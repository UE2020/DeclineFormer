pub mod token;
use anyhow::bail;
use rand::seq::SliceRandom;
use std::{fs::{read_to_string, remove_dir_all}, time::{Instant, Duration}};
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
    src_tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    tgt_tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
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
        &src_tokenizer
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
    let max_len = num_tokens * 2;
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
    Ok(tgt_tokenizer.decode(&tokens, false).unwrap())
}

fn main() -> Result<(), anyhow::Error> {
    let args = std::env::args().collect::<Vec<_>>();
    let mut masker = CModule::load("mask.pt")?;
    masker.set_eval();
    let device = Device::cuda_if_available();
    match args[1].as_str() {
        "test-tok" => {
            let tokenizer = token::load(&args[2]).expect("failed to train tokenizer");
            println!("{:?}", tokenizer.encode(args.get(3).map(|s| s.as_str()).unwrap_or("in<SEP><ABL>beginning<SEP>it created<SEP><NOM>God<SEP><ACC>heaven,<SEP>and<SEP><ACC>earth."), false).unwrap().get_tokens());
        }
        "train" => {
            remove_dir_all("./logdir").ok();
            let mut train_writer =
                tensorboard::summary_writer::SummaryWriter::new("./logdir/train");
            const BATCH_SIZE: usize = 128;
            let src_tokenizer = token::train_tokenizer(&args[2], "src_tokenizer.json", args[4].parse()?)
                .expect("failed to train & save tokenizer");
            let tgt_tokenizer = token::train_tokenizer(&args[3], "tgt_tokenizer.json", args[4].parse()?)
                .expect("failed to train & save tokenizer");
            let vs = VarStore::new(device);
            let mut net = TrainableCModule::load("init.pt", vs.root())?;
            net.set_train();
            let mut opt = nn::Adam::default()
                .beta1(0.9)
                .beta2(0.98)
                .eps(1e-9)
                .build(&vs, 0.0001)?;
            let file = read_to_string(&args[5])?;
            let flip = args[6] == "true";
            let hours: f32 = args[7].parse()?;
            let mut pairs: Vec<[&str; 2]> = file
                .lines()
                .map(|l| {
                    let split: Vec<_> = l.split('\t').collect();
                    if flip {
                        [split[1].trim(), split[0].trim()]
                    } else {
                        [split[0].trim(), split[1].trim()]
                    }
                })
                .collect();
            println!("Data is in memory.");
            let loss = |t: Tensor, target: Tensor| {
                t.cross_entropy_loss::<Tensor>(&target, None, Reduction::Mean, PAD_IDX, 0.0)
            };
            let mut steps = 0;
            let now = Instant::now();
            'outer: for epoch in 1.. {
                pairs.shuffle(&mut rand::thread_rng());
                for batch in pairs.chunks(BATCH_SIZE) {
                    if batch.len() < BATCH_SIZE {
                        continue;
                    }
                    let mut src_batch = vec![];
                    let mut tgt_batch = vec![];
                    for [src_sample, tgt_sample] in batch {
                        src_batch.push(tensor_transform(
                            &src_tokenizer
                                .encode(*src_sample, true)
                                .unwrap()
                                .get_ids()
                                .into_iter()
                                .map(|id| *id as i64)
                                .collect::<Vec<_>>(),
                        ));
                        tgt_batch.push(tensor_transform(
                            &tgt_tokenizer
                                .encode(*tgt_sample, true)
                                .unwrap()
                                .get_ids()
                                .into_iter()
                                .map(|id| *id as i64)
                                .collect::<Vec<_>>(),
                        ));
                    }
                    let src = Tensor::pad_sequence::<Tensor>(&src_batch, false, PAD_IDX as f64);
                    let tgt = Tensor::pad_sequence::<Tensor>(&tgt_batch, false, PAD_IDX as f64);
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
                            src.to_device(device),
                            tgt_input.to_device(device),
                            src_mask.to_device(device),
                            tgt_mask.to_kind(Kind::Bool).to_device(device),
                            src_padding_mask.shallow_clone().to_device(device),
                            tgt_padding_mask.to_device(device),
                            src_padding_mask.to_device(device),
                        ],
                    )?;
                    opt.zero_grad();
                    let tgt_out = &tgt.narrow(0, 1, tgt.size()[0] - 1).to_device(device);
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
                        let pair = pairs.choose(&mut rand::thread_rng()).expect("failed to sample dataset");
                        println!("input:        {}", pair[0]);
                        println!("ground truth: {}", pair[1]);
                        println!("sample:       {}", greedy_decode(pair[0], &net, &masker, &src_tokenizer, &tgt_tokenizer, device)?);
                        net.set_train();
                    }
                    if now.elapsed() >= Duration::from_secs_f32(hours * 3600.0) {
                        break 'outer;
                    }
                }
                println!("Epoch {} complete", epoch);
                net.save(&format!("model_{}.pt", epoch))?;
            }
            net.save("final.pt")?;
        }
        "test" => {
            let vs = VarStore::new(device);
            let mut net = TrainableCModule::load(&args[2], vs.root())?;
            net.set_eval();
            let src_tokenizer = token::load("src_tokenizer.json").unwrap();
            let tgt_tokenizer = token::load("tgt_tokenizer.json").unwrap();
            println!(
                "output: {}",
                greedy_decode(&args[3], &net, &masker, &src_tokenizer, &tgt_tokenizer, device)?
            );
        }
        _ => bail!("Invalid arguments"),
    }
    Ok(())
}
