pub mod batcher;
pub mod token;
use anyhow::bail;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use std::{
    fs::{read_to_string, remove_dir_all},
    time::{Duration, Instant},
};
use tch::{
    kind::Element,
    nn::{self, OptimizerConfig, VarStore},
    CModule, Device, IValue, IndexOp, Kind, Reduction, Tensor, TrainableCModule,
};
use tensorboard_rs as tensorboard;
use tinyvec::TinyVec;
use tokenizers::{Decoder, Model, Normalizer, PostProcessor, PreTokenizer, TokenizerImpl};

const HELP: &str = "seq2seq trainer: https://github.com/UE2020/DeclineFormer
    test: <model> <string>
    train: <src-corpus> <tgt-corpus> <vocab-size> <pair-corpus> <should-swap> <hours> <tgt-tokens-per-batch>
    test-tok: <tokenizer> <string>";

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

pub enum DecodeInput<'a> {
    Str(&'a str),
    Tokens(&'a [i64]),
}

pub fn greedy_decode<M, N, PT, PP, D>(
    input: DecodeInput,
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
    let src = tensor_transform(&match input {
        DecodeInput::Str(s) => src_tokenizer
            .encode(s, true)
            .unwrap()
            .get_ids()
            .into_iter()
            .map(|id| *id as i64)
            .collect::<Vec<_>>(),
        DecodeInput::Tokens(tokens) => tokens.to_vec(),
    })
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


pub fn beam_search<M, N, PT, PP, D>(
    input: DecodeInput,
    net: &TrainableCModule,
    masker: &CModule,
    src_tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    tgt_tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    device: Device,
    beam_size: usize,
    length_penalty: f64,
) -> Result<String, anyhow::Error>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    println!();
    let src = tensor_transform(&match input {
        DecodeInput::Str(s) => src_tokenizer
            .encode(s, true)
            .unwrap()
            .get_ids()
            .into_iter()
            .map(|id| *id as i64)
            .collect::<Vec<_>>(),
        DecodeInput::Tokens(tokens) => tokens.to_vec(),
    })
    .view((-1, 1))
    .to_device(device);
    let num_tokens = src.size()[0];
    let src_mask = Tensor::zeros(&[num_tokens, num_tokens], (Kind::Bool, device));
    let max_len = num_tokens * 2;
    let start_symbol = BOS_IDX;
    let memory = net.method_ts("encode", &[src, src_mask])?;
    let mut sequences = vec![vec![(start_symbol as i64, 0.0)]];
    for _ in 0..(max_len - 1) {
        let mut all_candidates = vec![];
        for sequence in &sequences {
            if sequence.last().unwrap().0 == (EOS_IDX as i64) {
                all_candidates.push(sequence.clone());
                continue;
            }
            let ys = Tensor::from_slice(&sequence.iter().map(|(token, _)| *token).collect::<Vec<_>>()).view((-1, 1)).to_device(device);
            let memory = memory.to_device(device);
            let tgt_mask = masker
                .method_ts("generate_square_subsequent_mask", &[ys.shallow_clone()])?
                .to_kind(Kind::Bool)
                .to_device(device);
            let out = net
                .method_ts("decode", &[ys.shallow_clone(), memory, tgt_mask])?
                .transpose(0, 1);
            let prob = net.method_ts("out_linear", &[out.i((.., -1))])?.log_softmax(-1, None);
            let (log_probs, next_words) = prob.topk(beam_size as i64, 1, true, true);
            for i in 0..beam_size {
                let token = i64::try_from(next_words.squeeze().i(i as i64))?;
                let log_prob = f64::try_from(log_probs.squeeze().i(i as i64))?;
                let mut new_sequence = sequence.clone();
                new_sequence.push((token as i64, log_prob));
                all_candidates.push(new_sequence);
            }
        }
        sequences = all_candidates
            .into_iter()
            .map(|sequence| {
                let len = sequence.len() as f64;
                let score = sequence.iter().map(|(_, log_prob)| log_prob).sum::<f64>();
                let norm = ((5.0 + len) / (5.0 + 1.0)).powf(length_penalty);
                let normalized_score = score / norm;
                (sequence.clone(), normalized_score)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .rev()
            .take(beam_size)
            .map(|(sequence, _)| sequence)
            .collect();
        //use std::io::Write;
        let s = sequences[0].iter().map(|(token, _)| *token as u32).collect::<Vec<_>>();
        print!("{esc}c", esc = 27 as char);
        println!("{}", tgt_tokenizer.decode(&s, false).unwrap());
        //std::io::stdout().flush().ok();
    }
    for sequence in sequences.iter() {
        let norm = ((5.0 + sequence.len() as f64) / (5.0 + 1.0)).powf(length_penalty);
        let score: f64 = sequence.iter().map(|(_, prob)| *prob).sum::<f64>() / norm;
        let s = sequence.iter().map(|(token, _)| *token as u32).collect::<Vec<_>>();
        //println!("{}: {}", score, tgt_tokenizer.decode(&s, false).unwrap());
    }
    let s = sequences[0].iter().map(|(token, _)| *token as u32).collect::<Vec<_>>();
    println!();
    Ok(tgt_tokenizer.decode(&s, false).unwrap())
}

fn up() -> String {
    format!("{}[A", ESC)
}

fn erase() -> String {
    format!("{}[2K", ESC)
}

const ESC: char = 27u8 as char;

pub fn flat_tensor_array<T: Element>(tensor: &Tensor) -> Result<Vec<T>, anyhow::Error> {
    let num_elem = tensor.numel();
    let mut vec = vec![T::ZERO; num_elem];
    tensor.f_to_kind(T::KIND)?.f_copy_data(&mut vec, num_elem)?;
    let shape: Vec<usize> = tensor.size().iter().map(|s| *s as usize).collect();
    dbg!(shape);
    Ok(vec)
}

fn get_learning_rate(step: usize, embed_dim: usize, warmup_steps: usize) -> f64 {
    return (embed_dim as f64).powf(-0.5)
        * ((step as f64).powf(-0.5)).min(step as f64 * (warmup_steps as f64).powf(-1.5));
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
            let mut test_writer = tensorboard::summary_writer::SummaryWriter::new("./logdir/test");
            let src_tokenizer =
                token::train_tokenizer(&args[2], "src_tokenizer.json", args[4].parse()?)
                    .expect("failed to train & save tokenizer");
            let tgt_tokenizer =
                token::train_tokenizer(&args[3], "tgt_tokenizer.json", args[4].parse()?)
                    .expect("failed to train & save tokenizer");
            let vs = VarStore::new(device);
            let mut net = TrainableCModule::load("init.pt", vs.root())?;
            net.set_train();
            let mut opt = nn::Adam::default()
                .beta1(0.9)
                .beta2(0.98)
                .eps(1e-9)
                .build(&vs, 0.0005)?;
            let file = read_to_string(&args[5])?;
            let flip = args[6] == "true";
            let hours: f32 = args[7].parse()?;
            let tgt_tokens: usize = args[8].parse()?;
            let mut train_pairs: Vec<[TinyVec<[i64; 128]>; 2]> = file
                .lines()
                .map(|l| {
                    let split: Vec<_> = l.split('\t').collect();
                    split
                })
                .filter(|split| split.len() == 2)
                .map(|split| {
                    if flip {
                        [split[1].trim(), split[0].trim()]
                    } else {
                        [split[0].trim(), split[1].trim()]
                    }
                })
                //.filter(|split| split[0].len() < (140 * 3))
                .map(|split| {
                    [
                        src_tokenizer
                            .encode(split[0], true)
                            .unwrap()
                            .get_ids()
                            .into_iter()
                            .map(|id| *id as i64)
                            .collect::<TinyVec<_>>(),
                        tgt_tokenizer
                            .encode(split[1], true)
                            .unwrap()
                            .get_ids()
                            .into_iter()
                            .map(|id| *id as i64)
                            .collect::<TinyVec<_>>(),
                    ]
                })
                .collect();
            let len = train_pairs.len();
            println!("{} pairs loaded", len);
            // ensure that the test pairs are randomly sampled
            train_pairs.shuffle(&mut rand::thread_rng());
            let mut test_pairs = train_pairs.split_off(len - 2500);
            train_pairs.sort_by_key(|[_, tgt]| tgt.len());
            test_pairs.sort_by_key(|[_, tgt]| tgt.len());
            println!("Data is in memory.");
            let loss = |t: Tensor, target: Tensor, label_smoothing: f64| {
                t.cross_entropy_loss::<Tensor>(
                    &target,
                    None,
                    Reduction::Mean,
                    PAD_IDX,
                    label_smoothing,
                )
            };
            let mut steps = 0;
            let now = Instant::now();
            let accum_iter = 10000 / tgt_tokens;
            'outer: for epoch in 1.. {
                opt.zero_grad();
                let mut total_loss = 0.0;
                let mut epoch_steps = 0;
                let mut epoch_updates = 0;
                let batches = batcher::batch(&train_pairs, tgt_tokens);
                for batch in batches {
                    steps += 1;
                    epoch_steps += 1;
                    let mut src_batch = vec![];
                    let mut tgt_batch = vec![];
                    for [src_sample, tgt_sample] in batch {
                        src_batch.push(tensor_transform(&src_sample));
                        tgt_batch.push(tensor_transform(&tgt_sample));
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
                    let src_padding_mask = src_padding_mask.to_device(device);
                    let logits = net.method_ts(
                        "forward",
                        &[
                            src.to_device(device),
                            tgt_input.to_device(device),
                            src_mask.to_device(device),
                            tgt_mask.to_kind(Kind::Bool).to_device(device),
                            src_padding_mask.shallow_clone(),
                            tgt_padding_mask.to_device(device),
                            src_padding_mask,
                        ],
                    )?;
                    let tgt_out = &tgt.narrow(0, 1, tgt.size()[0] - 1).to_device(device);
                    let logits_shape = logits.size();
                    let loss = loss(
                        logits.reshape(&[-1i64, logits_shape[logits_shape.len() - 1]]),
                        tgt_out.reshape(&[-1]),
                        0.1,
                    );
                    (&loss / accum_iter as f64).backward();
                    if epoch_steps % accum_iter == 0 {
                        epoch_updates += 1;
                        opt.set_lr(get_learning_rate(epoch_updates, 256, 4000));
                        opt.step();
                        opt.zero_grad();
                    }
                    let loss = f32::try_from(loss)?;
                    total_loss += loss;
                    train_writer.add_scalar("Loss", loss, steps as _);
                    if steps % 1000 == 0 {
                        net.set_eval();
                        let pair = train_pairs
                            .choose(&mut rand::thread_rng())
                            .expect("failed to sample dataset");
                        println!(
                            "input:        {}",
                            src_tokenizer
                                .decode(
                                    &pair[0].iter().map(|s| *s as u32).collect::<Vec<_>>(),
                                    false
                                )
                                .unwrap()
                        );
                        println!(
                            "ground truth: {}",
                            tgt_tokenizer
                                .decode(
                                    &pair[1].iter().map(|s| *s as u32).collect::<Vec<_>>(),
                                    false
                                )
                                .unwrap()
                        );
                        println!(
                            "sample:       {}",
                            greedy_decode(
                                DecodeInput::Tokens(&pair[0]),
                                &net,
                                &masker,
                                &src_tokenizer,
                                &tgt_tokenizer,
                                device
                            )?
                        );
                        net.set_train();
                    }
                    if now.elapsed() >= Duration::from_secs_f32(hours * 3600.0) {
                        break 'outer;
                    }
                }

                let mut test_loss_total = 0.0;
                let mut test_steps = 0;
                let batches = batcher::batch(&test_pairs, tgt_tokens);
                for batch in batches {
                    test_steps += 1;
                    let mut src_batch = vec![];
                    let mut tgt_batch = vec![];
                    for [src_sample, tgt_sample] in batch {
                        src_batch.push(tensor_transform(&src_sample));
                        tgt_batch.push(tensor_transform(&tgt_sample));
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
                    let src_padding_mask = src_padding_mask.to_device(device);
                    let logits = net.method_ts(
                        "forward",
                        &[
                            src.to_device(device),
                            tgt_input.to_device(device),
                            src_mask.to_device(device),
                            tgt_mask.to_kind(Kind::Bool).to_device(device),
                            src_padding_mask.shallow_clone(),
                            tgt_padding_mask.to_device(device),
                            src_padding_mask,
                        ],
                    )?;
                    let tgt_out = &tgt.narrow(0, 1, tgt.size()[0] - 1).to_device(device);
                    let logits_shape = logits.size();
                    let loss = loss(
                        logits.reshape(&[-1i64, logits_shape[logits_shape.len() - 1]]),
                        tgt_out.reshape(&[-1]),
                        0.0,
                    );
                    let loss = f32::try_from(loss)?;
                    test_loss_total += loss;
                }
                test_writer.add_scalar("Loss", test_loss_total / test_steps as f32, steps as _);
                println!(
                    "---------------------------------------------------------------------------"
                );
                println!("Epoch {} complete\ntrain loss={:.2}\ttrain PPL={:.4}\ntest loss={:.2}\ttest PPL={:.4}", epoch, total_loss / epoch_steps as f32, (total_loss / epoch_steps as f32).exp(), test_loss_total / test_steps as f32, (test_loss_total / test_steps as f32).exp());
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
            tch::no_grad(|| {
                println!(
                    "output: {}\nbeam  : {}",
                    greedy_decode(
                        DecodeInput::Str(&args[3]),
                        &net,
                        &masker,
                        &src_tokenizer,
                        &tgt_tokenizer,
                        device,
                        //4
                    ).unwrap(),
                    beam_search(
                        DecodeInput::Str(&args[3]),
                        &net,
                        &masker,
                        &src_tokenizer,
                        &tgt_tokenizer,
                        device,
                        4,
                        0.6,
                    ).unwrap()
                );
            });

        }
        "help" => println!("{}", HELP),
        _ => bail!("Invalid arguments\n{}", HELP),
    }
    Ok(())
}
