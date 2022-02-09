def embedding(vectorInput,embedding_matrix):
  if len(vectorInput)==1:
      return embedding_matrix[vectorInput[0]]
  else:
      return ivy.concatenate([embedding_matrix[vectorInput[0]], embedding(vectorInput[1:], embedding_matrix)], 0)


def embeddingV2(vectorInput,vocabulary_size, output_dim, input_length):
  embedding_initial = ivy.random_uniform(shape = (vocabulary_size,output_dim))
  embedding_matrix = ivy.variable(embedding_initial)
  vectorInput = ivy.squeeze(vectorInput)
  result = embedding(vectorInput, embedding_matrix)
  return ivy.reshape(result, [input_length, output_dim])

class UniformEmbeddingivy(ivy.Module):

  def __init__(self, vocabulary_size, output_dim):
    self.vocabulary_size = vocabulary_size
    self.output_dim = output_dim
    self.embedding_initial = ivy.random_uniform(shape = (self.vocabulary_size,self.output_dim))
    super().__init__()

  def _create_variables(self, dev):
    self.embedding_matrix = ivy.variable(self.embedding_initial)
    return {'embedding_matrix':self.embedding_matrix}

  def _forward(self,x):
    x = ivy.squeeze(x)
    return embeddingV2(x,self.vocabulary_size, self.output_dim, len(x))


##torch embedding subclass##

class UniformEmbedding(nn.Module):
  def __init__(self, vocabulary_size, output_dim):
    super().__init__()
    self.vocabulary_size = vocabulary_size
    self.output_dim = output_dim
    #self.input_legth = input_length
    self.embedding_initial = ivy.random_uniform(shape = (vocabulary_size,output_dim))
    self.embedding_matrix = nn.Parameter(self.embedding_initial, requires_grad = True)


  def forward(self,x):
    x = ivy.squeeze(x)
    return embeddingV2(x,self.vocabulary_size, self.output_dim, len(x))
