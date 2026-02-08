local Tensor = require("luanet.tensor")

local Serialization = {}

function Serialization.save(model, filename)
    local state = model:state_dict()
    local file = io.open(filename, "w")
    if not file then error("Could not open file " .. filename) end

    file:write("local luanet = require('luanet.init')\n")
    file:write("return {\n")

    for key, tensor in pairs(state) do
        file:write("['" .. key .. "'] = luanet.Tensor.new({")
        local data = tensor.data
        for i = 1, #data do
            file:write(string.format("%.8g", data[i]))
            if i < #data then file:write(",") end
        end
        file:write("}, {")
        local shape = tensor.shape
        for i = 1, #shape do
            file:write(shape[i])
            if i < #shape then file:write(",") end
        end
        file:write("}),\n")
    end

    file:write("}\n")
    file:close()
end

function Serialization.load(filename)
    local chunk, err = loadfile(filename)
    if not chunk then error("Could not load file: " .. err) end
    return chunk()
end

return Serialization
