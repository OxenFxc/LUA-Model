local Dataset = {}
Dataset.__index = Dataset

function Dataset:new()
    local o = setmetatable({}, self)
    return o
end

function Dataset:__len()
    error("Not implemented")
end

function Dataset:__getitem(idx)
    error("Not implemented")
end

return Dataset
