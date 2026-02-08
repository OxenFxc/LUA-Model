local layers = {}

layers.Linear = require("luanet.layers.linear")
layers.Sequential = require("luanet.layers.sequential")
layers.Embedding = require("luanet.layers.embedding")
layers.LayerNorm = require("luanet.layers.layernorm")
layers.MultiHeadAttention = require("luanet.layers.attention")
layers.Dropout = require("luanet.layers.dropout")

local activations = require("luanet.layers.activations")
layers.ReLU = activations.ReLU
layers.Sigmoid = activations.Sigmoid
layers.Tanh = activations.Tanh
layers.Softmax = activations.Softmax
layers.GELU = activations.GELU

return layers
