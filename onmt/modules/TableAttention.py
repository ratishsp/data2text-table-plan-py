import torch
import torch.nn as nn
from torch.autograd import Variable

from onmt.Utils import aeq, sequence_mask


class TableAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """
    def __init__(self, dim, coverage=False, attn_type="dot"):
        super(TableAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["general"]), (
                "Please select a valid attention type.")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)

        self.sm = nn.Softmax(-1)
        self.tt = torch.cuda if torch.cuda.is_available() else torch
    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)

    def forward(self, input, memory_bank, memory_lengths=None, stage1_target=None, plan_attn=None,
                player_row_indices=None, team_row_indices=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """
        PLAYER_ROWS = 26
        TEAM_ROWS = 2
        EXTRA_RECORDS = 4
        PLAYER_COLS = 22
        TEAM_COLS = 15
        PLAYER_RECORDS_MAX=EXTRA_RECORDS+PLAYER_ROWS*PLAYER_COLS

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = memory_bank.size()
        #print 'batch, sourceL, dim',batch, sourceL, dim
        batch_, targetL, dim_ = input.size()
        #print 'batch_, targetL, dim_',batch_, targetL, dim_
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        SOURCEL = sourceL
        targetL_st1_tgt, batch_st1_tgt,_= stage1_target.size()
        batch_plan, target_plan = plan_attn.size()
        aeq(batch_plan, batch)
        aeq(batch_plan, batch_st1_tgt)
        aeq(target_plan, targetL_st1_tgt)

        target_player_indices_L, batch_player_ind, player_rows_len = player_row_indices.size()
        aeq(target_player_indices_L, targetL_st1_tgt)
        aeq(batch_player_ind, batch)
        aeq(player_rows_len, PLAYER_ROWS)

        target_team_indices_L, batch_team_ind, team_rows_len = team_row_indices.size()
        aeq(target_team_indices_L, targetL_st1_tgt)
        aeq(batch_team_ind, batch)
        aeq(team_rows_len, TEAM_ROWS)

        # compute attention scores, as in Luong et al.
        align = self.score(input, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align = align.view(batch * targetL, sourceL)
        align_player_cells = self.sm(align[:,EXTRA_RECORDS:PLAYER_RECORDS_MAX].contiguous().view(-1, PLAYER_COLS))
        align_team_cells = self.sm(align[:,PLAYER_RECORDS_MAX:SOURCEL].contiguous().view(-1, TEAM_COLS))
        row_indices = (stage1_target.data.squeeze(2)-EXTRA_RECORDS)/PLAYER_COLS
        prob_prod = plan_attn.t() * Variable(row_indices.lt(PLAYER_ROWS).float(), requires_grad=False)    #stores probabilities for player records
        # (batch, 1, t_len_plan) x (batch, t_len_plan, 26) --> (batch, 1, 26)
        player_prob = torch.bmm(prob_prod.t().unsqueeze(1), player_row_indices.transpose(0,1).float()).squeeze(1)
        player_prob = player_prob.unsqueeze(2).expand(-1,-1,PLAYER_COLS).contiguous().view(-1,PLAYER_COLS)
        player_prob_table = align_player_cells*player_prob

        prob_prod = plan_attn.t() * Variable(row_indices.ge(PLAYER_ROWS).float(), requires_grad=False)    #stores probabilities for team records
        # (batch, 1, t_len_plan) x (batch, t_len_plan, 2) --> (batch, 1, 2)
        team_prob = torch.bmm(prob_prod.t().unsqueeze(1), team_row_indices.transpose(0,1).float()).squeeze(1)
        team_prob = team_prob.unsqueeze(2).expand(-1,-1,TEAM_COLS).contiguous().view(-1,TEAM_COLS)
        team_prob_table = align_team_cells*team_prob

        extra_prob_table = Variable(self.tt.FloatTensor(batch, EXTRA_RECORDS).fill_(0), requires_grad=False)
        align_vectors = torch.cat([extra_prob_table, player_prob_table.view(batch,-1), team_prob_table.view(batch,-1)],1)
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        batch_table, sourceL_table, dim_table = memory_bank.size()
        aeq(batch, batch_table)
        aeq(dim, dim_table)


        if one_step:
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        return align_vectors
