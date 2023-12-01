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
