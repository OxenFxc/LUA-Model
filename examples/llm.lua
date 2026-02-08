local luanet = require("luanet.init")
local Tensor = require("luanet.tensor")
local layers = require("luanet.layers.init")
local optim = require("luanet.optim.init")
local loss = require("luanet.loss.init")
local models = require("luanet.models.llm")
local utils = require("luanet.utils.init")
local data = require("luanet.data.init")

-- Define Dataset
local SeqDataset = setmetatable({}, {__index = data.Dataset})
SeqDataset.__index = SeqDataset

function SeqDataset:new(data_seq, seq_len)
    local self = setmetatable({}, SeqDataset)
    self.data = data_seq
    self.seq_len = seq_len
    return self
end

function SeqDataset:len()
    -- Number of possible sequences? Or just fixed samples?
    -- Example logic was:
    -- for b=1, batch_size do ... end
    -- effectively infinite or just repetitive.
    -- Let's define a fixed number of samples for the epoch.
    return 100
end

function SeqDataset:getitem(idx)
    -- Ignore idx, just generate the pattern 1, 2, 3...
    -- Wait, for training to work well, we want consistent data?
    -- The previous example used fixed `input` tensor.
    -- Let's make it consistent.
    -- Pattern: 1, 2, 3, 1, 2, 3 ...
    -- input: 1..seq_len
    -- target: 2..seq_len+1

    local d = self.data
    local seq = {}
    local target = {}

    -- Random start or fixed? Let's use random start to robustness?
    -- Or fixed pattern to ensure convergence quickly for this toy example.
    -- Let's just return the same sequence every time: 1, 2, 3, 1, 2, 3

    local start = 0
    for i = 1, self.seq_len do
        local val = d[(start + i - 1) % #d + 1]
        table.insert(seq, val)
        local next_val = d[(start + i) % #d + 1]
        table.insert(target, next_val)
    end

    -- Return tensors
    return {
        input = Tensor.new(seq, {self.seq_len}),
        target = Tensor.new(target, {self.seq_len})
    }
end

local vocab_size = 5
local embed_dim = 16
local num_heads = 4
local num_layers = 2
local dropout_p = 0.1

local model = models.MiniLLM:new(vocab_size, embed_dim, num_heads, num_layers, dropout_p)
local criterion = loss.CrossEntropyLoss:new()

local params, grads = model:parameters()
local optimizer = optim.Adam:new(params, grads, 0.01)

-- Data
local seq_len = 6
local batch_size = 4
local raw_data = {1, 2, 3}

local dataset = SeqDataset:new(raw_data, seq_len)
local dataloader = data.DataLoader:new(dataset, batch_size, true)

print("Training LLM with DataLoader, GELU, and Gradient Clipping...")
math.randomseed(1234)

model:train()
for epoch = 1, 5 do
    local step = 0
    local total_loss = 0

    -- Iterator
    local iter = dataloader:iterator()
    while true do
        local batch = iter()
        if not batch then break end

        -- batch is {input = Tensor(B, S), target = Tensor(B, S)}
        local input = batch.input
        local target = batch.target

        optimizer:zero_grad()

        local logits = model:forward(input)
        local logits_flat = logits:view({input.shape[1] * input.shape[2], vocab_size})
        local target_flat = target:view({input.shape[1] * input.shape[2]})

        local l = criterion:forward(logits_flat, target_flat)
        local grad_loss = criterion:backward()

        local grad_output = grad_loss:view({input.shape[1], input.shape[2], vocab_size})

        model:backward(grad_output)

        -- Clip gradients
        utils.clip_grad_norm(grads, 1.0)

        optimizer:step()

        total_loss = total_loss + l
        step = step + 1
    end

    print(string.format("Epoch %d, Avg Loss: %.4f", epoch, total_loss / step))
end

print("Saving model...")
utils.save(model, "llm_model.t7")

print("Loading model...")
local loaded_state = utils.load("llm_model.t7")
local new_model = models.MiniLLM:new(vocab_size, embed_dim, num_heads, num_layers, dropout_p)
new_model:load_state_dict(loaded_state)
new_model:eval()

-- Generation
print("Generating...")
local start_token = 1
local gen_seq = {start_token}
local max_len = 12

for i = 1, max_len do
    local inp = Tensor.new(gen_seq, {1, #gen_seq})
    local logits = new_model:forward(inp)
    local last_idx = #gen_seq
    local max_val = -math.huge
    local max_idx = 1

    for v = 1, vocab_size do
        local val = logits:get({1, last_idx, v})
        if val > max_val then
            max_val = val
            max_idx = v
        end
    end

    table.insert(gen_seq, max_idx)
end

print("Generated sequence: " .. table.concat(gen_seq, ", "))
