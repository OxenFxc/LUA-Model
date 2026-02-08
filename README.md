# Luanet

Luanet is a pure Lua neural network framework, inspired by PyTorch, designed for simplicity and educational purposes.

## Features

- **Tensors**: Multi-dimensional arrays with broadcasting, basic arithmetic operations, and matrix multiplication.
- **Autograd**: (Manual backward pass currently implemented in layers).
- **Layers**:
  - `Linear`, `Conv`, `RNN`, `Transformer`, `Embedding`, `LayerNorm`, `Dropout`, `GELU`.
- **Optimizers**:
  - `SGD`, `Adam`.
- **Loss Functions**:
  - `MSELoss`, `CrossEntropyLoss`.
- **Data Loaders**:
  - `Dataset`, `DataLoader` for batching and shuffling.
- **Models**:
  - Example implementations including a simple XOR solver and a mini LLM (Language Model).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/luanet.git
   cd luanet
   ```

2. Ensure you have Lua 5.1+ or LuaJIT installed on your system.

   - **macOS (Homebrew):** `brew install lua`
   - **Ubuntu/Debian:** `sudo apt-get install lua5.1` (or `luajit`)
   - **Windows:** Download from [Lua Binaries](http://luabinaries.sourceforge.net/)

   *Note: If you have the Lua source code in the `lua/` directory (not included by default), you can build it with `cd lua && make all`.*

## Usage

### Running Examples

To run the examples provided in the `examples/` directory:

1. **XOR Problem**:
   Solves the XOR classification problem using a simple feedforward neural network.
   ```bash
   lua examples/xor.lua
   ```

2. **Mini LLM**:
   Trains a small Transformer-based language model on a synthetic sequence dataset.
   ```bash
   lua examples/llm.lua
   ```

### Running Tests

To run the unit tests for tensors and other components:

```bash
lua tests/test_tensor.lua
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.
