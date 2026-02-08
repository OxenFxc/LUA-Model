local Tensor = {}
Tensor.__index = Tensor

local function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

local function calculate_strides(shape)
    local strides = {}
    local s = 1
    for i = #shape, 1, -1 do
        strides[i] = s
        s = s * shape[i]
    end
    return strides
end

local function get_size(shape)
    local s = 1
    for _, v in ipairs(shape) do
        s = s * v
    end
    return s
end

function Tensor.new(data, shape)
    local self = setmetatable({}, Tensor)
    if shape then
        self.shape = shape
        self.strides = calculate_strides(shape)
        self.data = data -- Expecting flat data if shape is provided
    else
        if type(data) == "table" then
            if data.data and data.shape then
                return data:clone()
            end
            self.shape = {#data}
            self.strides = {1}
            self.data = data
        else
             error("Invalid data for Tensor.new")
        end
    end
    return self
end

function Tensor.zeros(shape)
    local size = get_size(shape)
    local data = {}
    for i = 1, size do
        data[i] = 0
    end
    return Tensor.new(data, shape)
end

function Tensor.randn(shape)
    local size = get_size(shape)
    local data = {}
    for i = 1, size do
        local u1 = math.random()
        local u2 = math.random()
        local z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        data[i] = z0
    end
    return Tensor.new(data, shape)
end

function Tensor:clone()
    local newData = {}
    for i, v in ipairs(self.data) do
        newData[i] = v
    end
    local newShape = {}
    for i, v in ipairs(self.shape) do
        newShape[i] = v
    end
    return Tensor.new(newData, newShape)
end

local function get_index(self, indices)
    local idx = 1
    for i = 1, #self.shape do
        idx = idx + (indices[i] - 1) * self.strides[i]
    end
    return idx
end

function Tensor:get(indices)
    return self.data[get_index(self, indices)]
end

function Tensor:set(indices, val)
    self.data[get_index(self, indices)] = val
end

function Tensor:zero()
    for i = 1, #self.data do
        self.data[i] = 0
    end
end

function Tensor:view(new_shape)
    local size = get_size(new_shape)
    if size ~= get_size(self.shape) then
        error("Shape mismatch in view")
    end
    local t = Tensor.new(self.data, new_shape)
    return t
end

-- Element-wise operations
local function element_wise_op(a, b, op)
    local resData = {}
    local b_is_tensor = type(b) == "table" and b.data

    if b_is_tensor then
        -- Broadcasting support for 2D
        local broadcast_type = 0 -- 0: none, 1: row broadcast (b is 1xN), 2: col broadcast (b is Mx1)

        if #a.shape == 2 then
             if #b.shape == 1 then
                 if a.shape[2] == b.shape[1] then broadcast_type = 1 end
             elseif #b.shape == 2 then
                 if a.shape[2] == b.shape[2] and b.shape[1] == 1 then
                     broadcast_type = 1
                 elseif a.shape[1] == b.shape[1] and b.shape[2] == 1 then
                     broadcast_type = 2
                 end
             end
        end

        if broadcast_type > 0 then
             for i = 1, a.shape[1] do
                 for j = 1, a.shape[2] do
                     local idx_a = (i-1)*a.strides[1] + (j-1)*a.strides[2] + 1
                     local idx_b
                     if broadcast_type == 1 then
                         -- b is (1, N) or (N). varies with j.
                         if #b.shape == 1 then
                             idx_b = (j-1)*b.strides[1] + 1
                         else
                             idx_b = (j-1)*b.strides[2] + 1
                         end
                     else -- type 2
                         -- b is (M, 1). varies with i.
                         idx_b = (i-1)*b.strides[1] + 1
                     end
                     resData[idx_a] = op(a.data[idx_a], b.data[idx_b])
                 end
             end
             return Tensor.new(resData, a.shape)
        else
            if #a.shape ~= #b.shape then
                 error("Tensor shape mismatch for element-wise op: " .. table.concat(a.shape, "x") .. " vs " .. table.concat(b.shape, "x"))
            end
            for k = 1, #a.shape do
                if a.shape[k] ~= b.shape[k] then
                    error("Tensor shape mismatch: dim " .. k)
                end
            end
            local size = #a.data
            for i = 1, size do
                resData[i] = op(a.data[i], b.data[i])
            end
            return Tensor.new(resData, a.shape)
        end
    else
        -- Scalar
        local size = #a.data
        for i = 1, size do
            resData[i] = op(a.data[i], b)
        end
        return Tensor.new(resData, a.shape)
    end
end

function Tensor:add(other)
    return element_wise_op(self, other, function(x, y) return x + y end)
end

function Tensor:sub(other)
    return element_wise_op(self, other, function(x, y) return x - y end)
end

function Tensor:mul(other)
    return element_wise_op(self, other, function(x, y) return x * y end)
end

function Tensor:div(other)
    return element_wise_op(self, other, function(x, y) return x / y end)
end

function Tensor:matmul(other)
    if #self.shape ~= 2 or #other.shape ~= 2 then
        error("Matmul only supports 2D tensors for now")
    end
    if self.shape[2] ~= other.shape[1] then
        error("Matmul shape mismatch: " .. self.shape[1] .. "x" .. self.shape[2] .. " vs " .. other.shape[1] .. "x" .. other.shape[2])
    end

    local M = self.shape[1]
    local K = self.shape[2]
    local N = other.shape[2]

    local res = Tensor.zeros({M, N})

    for i = 1, M do
        for j = 1, N do
            local sum = 0
            for k = 1, K do
                local v1 = self.data[(i-1)*self.strides[1] + (k-1)*self.strides[2] + 1]
                local v2 = other.data[(k-1)*other.strides[1] + (j-1)*other.strides[2] + 1]
                sum = sum + v1 * v2
            end
            res.data[(i-1)*res.strides[1] + (j-1)*res.strides[2] + 1] = sum
        end
    end
    return res
end

function Tensor:transpose(dim1, dim2)
    if #self.shape ~= 2 then error("Transpose only for 2D") end
    local new_shape = {self.shape[2], self.shape[1]}
    local res = Tensor.zeros(new_shape)

    for i = 1, self.shape[1] do
        for j = 1, self.shape[2] do
             local val = self.data[(i-1)*self.strides[1] + (j-1)*self.strides[2] + 1]
             res.data[(j-1)*res.strides[1] + (i-1)*res.strides[2] + 1] = val
        end
    end
    return res
end

function Tensor:apply(func)
    local newData = {}
    for i, v in ipairs(self.data) do
        newData[i] = func(v)
    end
    return Tensor.new(newData, self.shape)
end

function Tensor:exp()
    return self:apply(math.exp)
end

function Tensor:log()
    return self:apply(math.log)
end

function Tensor:sqrt()
    return self:apply(math.sqrt)
end

function Tensor:pow(n)
    return self:apply(function(x) return x^n end)
end

-- Concatenate tensors along a dimension
function Tensor.cat(tensors, dim)
    if #tensors == 0 then error("Empty tensor list for cat") end
    local base_shape = tensors[1].shape
    dim = dim or 1

    -- Check shapes
    for i = 2, #tensors do
        local s = tensors[i].shape
        if #s ~= #base_shape then error("Shape mismatch in cat") end
        for d = 1, #base_shape do
            if d ~= dim and s[d] ~= base_shape[d] then
                error("Shape mismatch in cat at dim " .. d)
            end
        end
    end

    local new_shape = {}
    for d = 1, #base_shape do
        if d == dim then
            local sum = 0
            for i = 1, #tensors do sum = sum + tensors[i].shape[dim] end
            new_shape[d] = sum
        else
            new_shape[d] = base_shape[d]
        end
    end

    local res = Tensor.zeros(new_shape)
    local res_data = res.data
    local res_strides = res.strides

    -- Only support 1D and 2D for now? Or generic?
    -- Generic copy is tricky without recursive loops or coordinate calculation.
    -- Let's do generic coordinate iteration.
    -- But iterate over result coordinates and map back to source?
    -- Or iterate over source and map to result.

    -- Optimized for 1D and 2D
    if #base_shape == 1 then
        local offset = 0
        for i = 1, #tensors do
            local t = tensors[i]
            for k = 1, #t.data do
                res_data[offset + k] = t.data[k]
            end
            offset = offset + #t.data
        end
    elseif #base_shape == 2 then
        if dim == 1 then
            -- Stack rows
            local offset = 0
            for k = 1, #tensors do
                local t = tensors[k]
                -- rows: t.shape[1]
                -- cols: t.shape[2] (same for all)
                -- fast copy because data is row-major
                for i = 1, #t.data do
                    res_data[offset + i] = t.data[i]
                end
                offset = offset + #t.data
            end
        elseif dim == 2 then
            -- Stack cols
            -- Interleave
            local rows = base_shape[1]
            local current_col_offset = 0

            for k = 1, #tensors do
                local t = tensors[k]
                local cols = t.shape[2]

                for r = 0, rows - 1 do
                    for c = 0, cols - 1 do
                        -- Source index
                        local src_idx = r * t.strides[1] + c * t.strides[2] + 1
                        -- Dest index
                        -- row r, col (current_col_offset + c)
                        local dst_idx = r * res_strides[1] + (current_col_offset + c) * res_strides[2] + 1
                        res_data[dst_idx] = t.data[src_idx]
                    end
                end
                current_col_offset = current_col_offset + cols
            end
        else
            error("Invalid dim for 2D cat")
        end
    else
        error("Cat only supported for 1D and 2D tensors for now")
    end

    return res
end

function Tensor:sum(dim)
    if not dim then
        local s = 0
        for _, v in ipairs(self.data) do
            s = s + v
        end
        return s
    end

    -- Sum over dim
    -- Support dim=1 (rows) or dim=2 (cols) for 2D
    if #self.shape ~= 2 then error("Sum(dim) only for 2D") end

    if dim == 1 then -- Reduce rows, result (1, N) or (N)
        local N = self.shape[2]
        local res = Tensor.zeros({1, N})
        for i = 1, self.shape[1] do
            for j = 1, N do
                local val = self.data[(i-1)*self.strides[1] + (j-1)*self.strides[2] + 1]
                res.data[j] = res.data[j] + val
            end
        end
        return res
    elseif dim == 2 then -- Reduce cols, result (M, 1) or (M)
        local M = self.shape[1]
        local res = Tensor.zeros({M, 1})
        for i = 1, M do
            for j = 1, self.shape[2] do
                local val = self.data[(i-1)*self.strides[1] + (j-1)*self.strides[2] + 1]
                res.data[i] = res.data[i] + val
            end
        end
        return res
    else
        error("Invalid dim")
    end
end

Tensor.__add = Tensor.add
Tensor.__sub = Tensor.sub
Tensor.__mul = Tensor.mul
Tensor.__div = Tensor.div

function Tensor:__tostring()
    if #self.shape == 1 then
        local s = "["
        for i = 1, #self.data do
            s = s .. string.format("%.4f", self.data[i])
            if i < #self.data then s = s .. ", " end
        end
        return s .. "]"
    elseif #self.shape == 2 then
        local s = "["
        for i = 1, self.shape[1] do
            s = s .. "\n  ["
            for j = 1, self.shape[2] do
                local idx = (i-1)*self.strides[1] + (j-1)*self.strides[2] + 1
                s = s .. string.format("%.4f", self.data[idx])
                if j < self.shape[2] then s = s .. ", " end
            end
            s = s .. "]"
        end
        return s .. "\n]"
    else
        return "Tensor with shape (" .. table.concat(self.shape, ", ") .. ")"
    end
end

return Tensor
