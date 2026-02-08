local Serialization = require("luanet.utils.serialization")
local GradClip = require("luanet.utils.grad_clip")

local Utils = {}
Utils.save = Serialization.save
Utils.load = Serialization.load
Utils.clip_grad_norm = GradClip.clip_grad_norm

return Utils
