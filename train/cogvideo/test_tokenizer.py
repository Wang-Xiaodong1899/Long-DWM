from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-fs/models/CogVideoX-2b", subfolder="tokenizer" 
)

special_tokens = {"additional_special_tokens": ["<action_start>", "<action_end>"]}
tokenizer.add_special_tokens(special_tokens)

text_encoder = T5EncoderModel.from_pretrained(
        "/root/autodl-fs/models/CogVideoX-2b", subfolder="text_encoder" 
)

text_encoder.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained('/root/autodl-fs/my_t5_encoder_tokenizer')

text_encoder.save_pretrained('/root/autodl-fs/my_t5_encoder_model')
