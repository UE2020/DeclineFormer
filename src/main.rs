pub mod token;
use anyhow::bail;
use rand::seq::SliceRandom;
use std::fs::{read_to_string, remove_dir_all};
use tch::{
    nn::{self, OptimizerConfig, VarStore},
    CModule, Device, IValue, IndexOp, Kind, Reduction, Tensor, TrainableCModule,
};
use tensorboard_rs as tensorboard;
use tokenizers::{Decoder, Model, Normalizer, PostProcessor, PreTokenizer, TokenizerImpl};

pub fn ivalue_to_tensor(ivalue: &IValue) -> Result<Tensor, anyhow::Error> {
    match ivalue {
        IValue::Tensor(t) => Ok(t.shallow_clone()),
        _ => bail!("Not a tensor"),
    }
}

const _UNK_IDX: i64 = 0;
const PAD_IDX: i64 = 1;
const BOS_IDX: i64 = 2;
const EOS_IDX: i64 = 3;

fn tensor_transform(tokens: &[i64]) -> Tensor {
    Tensor::cat(
        &[
            Tensor::from_slice(&[BOS_IDX]),
            Tensor::from_slice(tokens),
            Tensor::from_slice(&[EOS_IDX]),
        ],
        0,
    )
}

pub fn greedy_decode<M, N, PT, PP, D>(
    input: &str,
    net: &TrainableCModule,
    masker: &CModule,
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    device: Device,
) -> Result<String, anyhow::Error>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    let src = tensor_transform(
        &tokenizer
            .encode(input, true)
            .unwrap()
            .get_ids()
            .into_iter()
            .map(|id| *id as i64)
            .collect::<Vec<_>>(),
    )
    .view((-1, 1))
    .to_device(device);
    let num_tokens = src.size()[0];
    let src_mask = Tensor::zeros(&[num_tokens, num_tokens], (Kind::Bool, device));
    let max_len = num_tokens + 5;
    let start_symbol = BOS_IDX;
    let memory = net.method_ts("encode", &[src, src_mask])?;
    let mut ys = Tensor::full(&[1, 1], start_symbol, (Kind::Int64, device));
    let mut tokens = vec![start_symbol as u32];
    for _ in 0..(max_len - 1) {
        let memory = memory.to_device(device);
        let tgt_mask = masker
            .method_ts("generate_square_subsequent_mask", &[ys.shallow_clone()])?
            .to_kind(Kind::Bool)
            .to_device(device);
        let out = net
            .method_ts("decode", &[ys.shallow_clone(), memory, tgt_mask])?
            .transpose(0, 1);
        let prob = net.method_ts("out_linear", &[out.i((.., -1))])?;
        let (_, next_word) = prob.max_dim(1, false);
        let token = i64::try_from(next_word)?;
        ys = Tensor::cat(
            &[ys, Tensor::full(&[1, 1], token, (Kind::Int64, device))],
            0,
        );
        tokens.push(token as u32);
        if token == EOS_IDX {
            break;
        }
    }
    Ok(tokenizer.decode(&tokens, false).unwrap())
}

fn main() -> Result<(), anyhow::Error> {
    let args = std::env::args().collect::<Vec<_>>();
    let mut masker = CModule::load("mask.pt")?;
    masker.set_eval();
    let device = Device::cuda_if_available();
    match args[1].as_str() {
        "test-tok" => {
            let tokenizer = token::train_tokenizer(&args[2], args[3].parse()?)
                .expect("failed to train tokenizer");
            println!("{:?}", tokenizer.encode(args.get(4).map(|s| s.as_str()).unwrap_or("in<SEP><ABL>beginning<SEP>it created<SEP><NOM>God<SEP><ACC>heaven,<SEP>and<SEP><ACC>earth."), false).unwrap().get_tokens());
        }
        "train" => {
            remove_dir_all("./logdir").ok();
            let mut train_writer =
                tensorboard::summary_writer::SummaryWriter::new("./logdir/train");
            const BATCH_SIZE: usize = 128;
            let tokenizer = token::train_tokenizer(&args[2], args[3].parse()?)
                .expect("failed to train & save tokenizer");
            let vs = VarStore::new(device);
            let mut net = TrainableCModule::load("init.pt", vs.root())?;
            net.set_train();
            let mut opt = nn::Adam::default()
                .beta1(0.9)
                .beta2(0.98)
                .eps(1e-9)
                .build(&vs, 0.0001)?;
            let file = read_to_string(&args[4])?;
            let mut pairs: Vec<[&str; 2]> = file
                .lines()
                .map(|l| {
                    let split: Vec<_> = l.split('\t').collect();
                    [split[0].trim(), split[1].trim()]
                })
                .collect();
            let loss = |t: Tensor, target: Tensor| {
                t.cross_entropy_loss::<Tensor>(&target, None, Reduction::Mean, PAD_IDX, 0.0)
            };
            let mut steps = 0;
            for epoch in 1.. {
                pairs.shuffle(&mut rand::thread_rng());
                for batch in pairs.chunks(BATCH_SIZE) {
                    let mut src_batch = vec![];
                    let mut tgt_batch = vec![];
                    for [src_sample, tgt_sample] in batch {
                        src_batch.push(tensor_transform(
                            &tokenizer
                                .encode(*src_sample, true)
                                .unwrap()
                                .get_ids()
                                .into_iter()
                                .map(|id| *id as i64)
                                .collect::<Vec<_>>(),
                        ));
                        tgt_batch.push(tensor_transform(
                            &tokenizer
                                .encode(*tgt_sample, true)
                                .unwrap()
                                .get_ids()
                                .into_iter()
                                .map(|id| *id as i64)
                                .collect::<Vec<_>>(),
                        ));
                    }
                    let src = Tensor::pad_sequence::<Tensor>(&src_batch, false, PAD_IDX as f64)
                        .to_device(device);
                    let tgt = Tensor::pad_sequence::<Tensor>(&tgt_batch, false, PAD_IDX as f64)
                        .to_device(device);
                    let tgt_input = tgt.narrow(0, 0, tgt.size()[0] - 1);
                    let masks = masker.method_is(
                        "create_mask",
                        &[
                            IValue::Tensor(src.shallow_clone()),
                            IValue::Tensor(tgt_input.shallow_clone()),
                        ],
                    )?;
                    let (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask) = match masks {
                        IValue::Tuple(masks) => (
                            ivalue_to_tensor(&masks[0])?,
                            ivalue_to_tensor(&masks[1])?,
                            ivalue_to_tensor(&masks[2])?,
                            ivalue_to_tensor(&masks[3])?,
                        ),
                        _ => bail!("Invalid structure from masker"),
                    };
                    let logits = net.method_ts(
                        "forward",
                        &[
                            src,
                            tgt_input,
                            src_mask,
                            tgt_mask,
                            src_padding_mask.shallow_clone(),
                            tgt_padding_mask,
                            src_padding_mask,
                        ],
                    )?;
                    opt.zero_grad();
                    let tgt_out = &tgt.narrow(0, 1, tgt.size()[0] - 1);
                    let logits_shape = logits.size();
                    let loss = loss(
                        logits.reshape(&[-1i64, logits_shape[logits_shape.len() - 1]]),
                        tgt_out.reshape(&[-1]),
                    );
                    loss.backward();
                    opt.step();
                    steps += 1;
                    train_writer.add_scalar("Loss", f32::try_from(loss)?, steps as _);
                    if steps % 50 == 0 {
                        net.set_eval();
                        println!("sample: {}", greedy_decode("in<SEP><ABL>beginning<SEP><ACT>it created<SEP><NOM>God<SEP><ACC>heaven,<SEP>and<SEP><ACC>earth.", &net, &masker, &tokenizer, device)?);
                        net.set_train();
                    }
                }
                println!("Epoch {} complete", epoch);
                net.save(&format!("model_{}.pt", epoch))?;
            }
        }
        "test" => {
            let vs = VarStore::new(device);
            let mut net = TrainableCModule::load(&args[2], vs.root())?;
            net.set_eval();
            let tokenizer = token::load().unwrap();
            println!(
                "output: {}",
                greedy_decode(&args[3], &net, &masker, &tokenizer, device)?
            );
        }
        _ => bail!("Invalid arguments"),
    }
    Ok(())
}
