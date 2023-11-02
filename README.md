# baichuan-ptune-deepPrompt
baichuan p-tune v2 fine tune


``class PrefixEncoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * config.hidden_size * 2) #[seq layer hid 2]
        self.cfg=config
    def forward(self, prefix: torch.Tensor): # [batch seq]
        batch,seq=prefix.shape
        past_key_values = self.embedding(prefix)
        past_key_values=past_key_values.view(2,self.cfg.num_hidden_layers ,batch,self.cfg.num_attention_heads,seq,self.cfg.hidden_size//self.cfg.num_attention_heads)
        return past_key_values  #[2,layer,batch,head,seq,dim]``
