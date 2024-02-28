# 解决方案
## DualModel
该模型用于同时训练两个模型  
核心结构如下：
```python
class DualModel(nn.Module):
    def __init__(self, cs_model, cg_model):
        """实现模型的初始化"""

    def forward(self, cs_input, cg_input, use_dual, code_score, summ_score) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        """
        传入cs_input, cg_input，对原句的操作在collect_fn中处理
        use_dual表示是否使用对偶
        先构造两个模型单独训练的输入，传入模型获得分别训练的输出
        如果不使用对偶：
            直接输出两个模型的输出，将损失相加后梯度下降
        如果使用对偶：
            输出中要有实现对偶的成分：注意力矩阵，解码输出
            根据注意力矩阵，利用KL散度构造注意力损失
            根据编码输出的概率计算条件概率损失
            几个损失相加作为最终的损失
        """

```

## PLBART的输入
- input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`): 词汇表中输入序列标记的索引。默认情况下，如果您提供了填充，则填充将被忽略。索引可以用[`AutoTokenizer`] 或者 [`PLBartMultiTokenizer`]获得，详见[`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] [What are input IDs?](../glossary#input-ids) 
- attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*): 屏蔽，以避免对填充标记索引执行关注。屏蔽值在 `[0, 1]` 中选择 - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**. - 1 表示token没有被屏蔽 - 0 表示token被屏蔽 [What are attention masks?](../glossary#attention-mask)
- decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*): 词汇表中解码器输入序列词块的索引。索引可以用[`AutoTokenizer`] 或者 [`PLBartMultiTokenizer`]获得，详见[`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] [What are decoder input IDs?](../glossary#decoder-input-ids)  PLBart 使 用特定的语言 id 标记作为生成 `decoder_input_ids` 的起始标记。 例如，en_XX 为 50003，java 为 50001。如果past_key_values被使用，则只需输入最后一个 "解码器输入标识符"（见past_key_values`）。对于翻译和摘要训练，应提供 `decoder_input_ids`。 如果没有提供 "decoder_input_ids"，模型将通过将 "input_ids "向右移动来创建该张量，以效仿论文进行去噪预训练。
- decoder_attention_mask (: obj:*torch.LongTensor* of shape `(batch_size, target_sequence_length)`, *optional*): 默认操作：生成的张量会忽略 `decoder_input_ids` 中的 pad 标记。默认情况下还会使用屏蔽
- head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*): 使编码器中注意模块头的屏蔽。屏蔽值在"[0, 1]"中选择: - 1 表示头没有被屏蔽 - 0 表示头被屏蔽
- decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*): 使解码器中注意模块头的屏蔽。屏蔽值在"[0, 1]"中选择： - 1 indicates the head is **not masked**, - 0 indicates the head is **masked**. - 1 表示头没有被屏蔽 - 0 表示头被屏蔽
- cross_attn_head_mask (: obj:*torch.Tensor* of shape `(decoder_layers, decoder_attention_heads)`, *optional*): 使解码器中交叉注意模块头无效的屏蔽。屏蔽值在"[0, 1]"中选择. - 1 表示头没有被屏蔽 - 0 表示头被屏蔽
- encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*): Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, 是编码器最后一层输出端的隐藏状态序列。用于解码器的交叉注意。
- past_key_values (: obj:*tuple(tuple(torch.FloatTensor))*, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`): 包含预先计算的隐藏状态（自注意区 块和交叉注意区块中的键和值），可用于（参见 `past_key_values` 输入）加速顺序解码。如果使用 "past_key_values"，用户可以选择只输入形状为"(batch_size, 1) "的最后一个 "decoder_input_ids" （那些没有将其过去键值状态提供给此模型的 "decoder_input_ids"）， 而不是形状为"(batch_size, sequence_length) "的所有 "decoder_input_ids"。
- inputs_embeds (: obj:*torch.FloatTensor* of shape `(batch_size, sequence_length, hidden_size)`, *optional*): 可以选择直接传递嵌入表示法，而不是传递 `input_ids`。 如果你想更多地控制如何将 `input_ids` 索引转换为关联向量，而不是模型的内部嵌入查找矩阵，这将非常有用。 - decoder_inputs_embeds (: obj:*torch.FloatTensor* of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*): 您可以选择直接传递嵌入的表示，而不是传递 `decoder_input_ids`。如果使用 "past_key_values"， 则只需输入最后的 "decoder_inputs_embeds"（参见 "past_key_values"）。 如果希望比模型的内部嵌入查找矩阵更能控制如何将 `decoder_input_ids` 索引转换为相关向量，这一点非常有用。如果 `decoder_input_ids` 和 `decoder_inputs_embeds` 都未设置，则 `decoder_inputs_embeds` 取 `inputs_embeds` 的值。
- use_cache (`bool`, *optional*):  如果设置为 "true"，则会返回 "past_key_values "键值状态，可用于加快解码速度（参见 "past_key_values"）。
- output_attentions (`bool`, *optional*): 是否返回所有注意力层的注意力张量。详情请参阅返回张量下的 "注意力"。
- output_hidden_states (`bool`, *optional*): 是否返回所有层的隐藏状态。详情请参阅返回张量下的 "hidden_states"。
- return_dict (`bool`, *optional*): 是否返回 [`~utils.ModelOutput`]，而不是普通的元组。

## PLBART的默认配置
```python
PLBartConfig {
  "_name_or_path": "./pretrained_cache/plbart-base",
  "activation_dropout": 0.0,
  "activation_function": "gelu",
  "architectures": [
    "PLBartForConditionalGeneration"
  ],
  "attention_dropout": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": 0.0,
  "d_model": 768,
  "decoder_attention_heads": 12,
  "decoder_ffn_dim": 3072,
  "decoder_layerdrop": 0.0,
  "decoder_layers": 6,
  "dropout": 0.1,
  "encoder_attention_heads": 12,
  "encoder_ffn_dim": 3072,
  "encoder_layerdrop": 0.0,
  "encoder_layers": 6,
  "eos_token_id": 2,
  "forced_eos_token_id": 2,
  "init_std": 0.02,
  "is_encoder_decoder": true,
  "max_position_embeddings": 1024,
  "model_type": "plbart",
  "num_hidden_layers": 6,
  "pad_token_id": 1,
  "scale_embedding": true,
  "transformers_version": "4.34.1",
  "use_cache": true,
  "vocab_size": 50005
}
```

## PLBART的输出
```python
    @ staticmethod
    def run_base_model(input_text: Union[str, List[str]],
                       target_text: Union[str, List[str]],
                       tokenizer: PLBartTokenizer,
                       model: PLBartForConditionalGeneration):
        if isinstance(input_text, str):
            input_text = [input_text]
        if isinstance(target_text, str):
            target_text = [target_text]
        to_inp = tokenizer(input_text,
                           return_tensors='pt',
                           padding=True,
                           max_length=256,
                           truncation=True)
        label = tokenizer(text_target=target_text,
                          return_tensors='pt',
                          padding=True,
                          max_length=256,
                          truncation=True)['input_ids']
        to_inp['labels'] = label
        to_inp['output_attentions'] = True
        to_inp['return_dict'] = True
        return model(**to_inp)
```
在上述输入格式下，可以输出格式如下：
- 输出类型为$transformers.modeling_outputs.Seq2SeqLMOutput$，和字典类似，可以用$\.$获得成员值。
- loss: torch.Tensor，维度为0的常数，模型的偏差损失，可导
- logits: torch.Tensor[batch_size, max(tgt_len)+1, vocab_size]，每个输出token的概率分布，可以用来求条件概率分布。"+1"是特殊标记<eos>
- past_key_values：比较复杂，没看懂
- decoder_attentions：Tuple(len=num_layers, Tensor[batch_size, num_head, max(tgt_len)+1, max(tgt_len)+1])，解码器的注意力矩阵
- cross_attentions: Tuple(len=num_layers, Tensor[batch_size, num_head, max(tgt_len)+1, max(src_len)+1])，交叉注意力矩阵，可以用于求注意力损失
- encoder_last_hidden_state: NonType
- encoder_attentions:Tuple(len=num_layers, Tensor[batch_size, num_head, max(src_len)+1, max(src_len)+1])，编码器的注意力矩阵

