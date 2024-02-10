use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::normalizers::{Lowercase, StripAccents};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Result, Tokenizer, TokenizerBuilder, TokenizerImpl};

pub fn train_tokenizer(
    vocab: &str,
    save_dir: &str,
    vocab_size: usize,
) -> Result<TokenizerImpl<BPE, Sequence, ByteLevel, ByteLevel, ByteLevel>> {
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(0)
        .special_tokens(vec![
            AddedToken::from(String::from("<UNK>"), true),
            AddedToken::from(String::from("<PAD>"), true),
            AddedToken::from(String::from("<BOS>"), true),
            AddedToken::from(String::from("<EOS>"), true),
            /*AddedToken::from(String::from("<S>"), true),
            AddedToken::from(String::from("<C:ABL>"), true),
            AddedToken::from(String::from("<C:ACC>"), true),
            AddedToken::from(String::from("<C:D>"), true),
            AddedToken::from(String::from("<C:G>"), true),
            AddedToken::from(String::from("<C:L>"), true),
            AddedToken::from(String::from("<C:N>"), true),
            AddedToken::from(String::from("<C:V>"), true),
            AddedToken::from(String::from("<D:C>"), true),
            AddedToken::from(String::from("<D:P>"), true),
            AddedToken::from(String::from("<D:S>"), true),
            AddedToken::from(String::from("<F:ADJ>"), true),
            AddedToken::from(String::from("<F:ADV>"), true),
            AddedToken::from(String::from("<F:C>"), true),
            AddedToken::from(String::from("<F:I>"), true),
            AddedToken::from(String::from("<F:N>"), true),
            AddedToken::from(String::from("<F:NUM>"), true),
            AddedToken::from(String::from("<F:PAR>"), true),
            AddedToken::from(String::from("<F:PREP>"), true),
            AddedToken::from(String::from("<F:PRON>"), true),
            AddedToken::from(String::from("<F:S>"), true),
            AddedToken::from(String::from("<F:V>"), true),
            AddedToken::from(String::from("<G:C>"), true),
            AddedToken::from(String::from("<G:F>"), true),
            AddedToken::from(String::from("<G:M>"), true),
            AddedToken::from(String::from("<G:N>"), true),
            AddedToken::from(String::from("<M:IMP>"), true),
            AddedToken::from(String::from("<M:IND>"), true),
            AddedToken::from(String::from("<M:INF>"), true),
            AddedToken::from(String::from("<M:S>"), true),
            AddedToken::from(String::from("<N:A>"), true),
            AddedToken::from(String::from("<N:C>"), true),
            AddedToken::from(String::from("<N:D>"), true),
            AddedToken::from(String::from("<N:O>"), true),
            AddedToken::from(String::from("<P:F>"), true),
            AddedToken::from(String::from("<P:T>"), true),
            AddedToken::from(String::from("<T:F>"), true),
            AddedToken::from(String::from("<T:FPERF>"), true),
            AddedToken::from(String::from("<T:I>"), true),
            AddedToken::from(String::from("<T:PERF>"), true),
            AddedToken::from(String::from("<T:PLUP>"), true),
            AddedToken::from(String::from("<T:PRES>"), true),
            AddedToken::from(String::from("<V:A>"), true),
            AddedToken::from(String::from("<V:P>"), true),
            AddedToken::from(String::from("<PPL:1>"), true),
            AddedToken::from(String::from("<PPL:2>"), true),
            AddedToken::from(String::from("<PPL:3>"), true),*/
            // AddedToken::from(String::from("<SEP>"), true),
            // AddedToken::from(String::from("<NOM>"), true),
            // AddedToken::from(String::from("<GEN>"), true),
            // AddedToken::from(String::from("<DAT>"), true),
            // AddedToken::from(String::from("<ACC>"), true),
            // AddedToken::from(String::from("<ABL>"), true),
            // AddedToken::from(String::from("<LOC>"), true),
            // AddedToken::from(String::from("<VOC>"), true),
        ])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            StripAccents.into(),
            NFC.into(),
            Lowercase.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    let pretty = false;
    tokenizer
        .train_from_files(&mut trainer, vec![vocab.to_string()])?
        .save(save_dir, pretty)?;
    Ok(tokenizer)
}

pub fn load(dir: &str) -> Result<Tokenizer> {
    Tokenizer::from_file(dir)
}
