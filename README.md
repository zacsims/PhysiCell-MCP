# PhysiCell MCP Server

An MCP (Model Context Protocol) server that provides tools to interact with PhysiCell simulations, including creating projects, configuring parameters, running simulations, and analyzing results.

## Features

The PhysiCell MCP Server provides the following tools:

- **create_project**: Create new PhysiCell projects from templates
- **list_projects**: List available sample projects 
- **configure_simulation**: Modify simulation parameters (time, domain, etc.)
- **run_simulation**: Compile and run PhysiCell simulations with automatic error fixing
- **get_simulation_status**: Check the current status of a simulation
- **analyze_results**: Analyze simulation outputs (cell counts, positions, velocities)
- **fix_compilation_issues**: Automatically diagnose and fix compilation problems

## Prerequisites

1. **PhysiCell**: Install PhysiCell from https://github.com/MathCancer/PhysiCell
2. **Python 3.10+**: Required for the MCP server (Python 3.8 not sufficient due to MCP dependencies)
3. **GCC with OpenMP**: For compiling PhysiCell (automatically configured for macOS with Homebrew)
4. **uv**: Fast Python package manager (install from https://github.com/astral-sh/uv)

### macOS Setup (Recommended)
The server is optimized for macOS with Homebrew:
```bash
# Install required compiler
brew install gcc@12

# Ensure environment variable is set
export PHYSICELL_CPP=/opt/homebrew/bin/g++-12
echo 'export PHYSICELL_CPP=/opt/homebrew/bin/g++-12' >> ~/.zshrc
```

## Installation

### Method 1: Using pip with virtual environment (Recommended)

1. **Install Python 3.10** (if not already installed):
   ```bash
   # macOS with Homebrew
   brew install python@3.10
   ```

2. **Clone or download the PhysiCell MCP server files**

3. **Create virtual environment and install dependencies**:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

### Method 2: Using uv (Alternative)

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

3. Ensure PhysiCell is installed and properly configured

## Configuration

### For Claude Desktop

Add the following to your Claude Desktop configuration file:

**On macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**On Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**Recommended Configuration (using virtual environment directly):**
```json
{
  "mcpServers": {
    "physicell": {
      "command": "/path/to/PhysiCell-MCP/.venv/bin/python",
      "args": ["/path/to/PhysiCell-MCP/physicell_mcp_server.py", "--physicell-path", "/path/to/PhysiCell"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**Alternative using uv:**
```json
{
  "mcpServers": {
    "physicell": {
      "command": "uv",
      "args": ["run", "python", "/path/to/PhysiCell-MCP/physicell_mcp_server.py", "--physicell-path", "/path/to/PhysiCell"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**Replace the paths:**
- `/path/to/PhysiCell-MCP/` → Your actual PhysiCell MCP server directory
- `/path/to/PhysiCell` → Your PhysiCell installation directory

**Example for typical macOS setup:**
```json
{
  "mcpServers": {
    "physicell": {
      "command": "/Users/yourusername/PhysiCell-MCP/.venv/bin/python",
      "args": ["/Users/yourusername/PhysiCell-MCP/physicell_mcp_server.py", "--physicell-path", "/Users/yourusername/Downloads/PhysiCell"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## Usage Examples

Once the server is running and connected to Claude, you can use natural language to interact with PhysiCell:

### Creating a Project
- "Create a new PhysiCell project using the biorobots template"
- "List all available PhysiCell sample projects"

### Running Simulations
- "Run a PhysiCell simulation with 8 threads"
- "Compile and run the current project"

### Troubleshooting
- "Fix any compilation issues with PhysiCell"
- "Check if any PhysiCell simulations are currently running"

### Analyzing Results
- "Analyze the cell count in the output directory"
- "Show me what output files were generated"

## Server Arguments

- `--physicell-path`: Path to your PhysiCell installation directory (default: `./PhysiCell`)

## Development

### Running Standalone

You can test the server standalone:

```bash
.venv/bin/python physicell_mcp_server.py --physicell-path /path/to/PhysiCell --help
```

### Testing Server Functionality

```bash
# Test server initialization
echo '{}' | .venv/bin/python physicell_mcp_server.py --physicell-path /path/to/PhysiCell
```

### Adding New Tools

To add new tools, modify the `list_tools()` method to define the tool schema and implement the corresponding handler in `call_tool()`.

## Troubleshooting

### Common Issues and Solutions

1. **"MCP physicell: Unexpected token 'H', "Hello from"... is not valid JSON"**
   - **Solution**: Make sure you're using `physicell_mcp_server.py` not `main.py` in your configuration

2. **"MCP physicell: Server disconnected"**
   - **Solution**: Check that the Python path and script path in your config are correct
   - **Check**: Verify the virtual environment exists: `ls /path/to/.venv/bin/python`

3. **Python version errors ("requires Python>=3.10")**
   - **Solution**: Install Python 3.10 or higher and recreate the virtual environment

4. **Compilation errors with architecture flags**
   - **Automatic fix**: Use the `fix_compilation_issues` tool through Claude
   - **Manual fix**: The server automatically detects and fixes `apple-m1` → `armv8-a` architecture issues

5. **PhysiCell not found**
   - **Solution**: Ensure the `--physicell-path` argument points to your PhysiCell installation
   - **Check**: Verify directory contains `Makefile`, `BioFVM/`, `core/`, etc.

6. **Compiler errors (OpenMP not found)**
   - **macOS Solution**: 
     ```bash
     brew install gcc@12
     export PHYSICELL_CPP=/opt/homebrew/bin/g++-12
     ```

7. **Permission errors**
   - **Solution**: Ensure you have write permissions in the PhysiCell directory

8. **Module not found errors**
   - **Solution**: Activate virtual environment and reinstall: `pip install -e .`

### Debug Steps

1. **Test MCP Server**:
   ```bash
   /path/to/.venv/bin/python /path/to/physicell_mcp_server.py --help
   ```

2. **Test PhysiCell Compilation**:
   ```bash
   cd /path/to/PhysiCell
   export PHYSICELL_CPP=/opt/homebrew/bin/g++-12  # macOS
   make clean && make -j4
   ```

3. **Check Configuration File**:
   - Verify JSON syntax is valid
   - Check that all paths exist and are accessible
   - Restart Claude Desktop after config changes

## Architecture Support

The server includes automatic fixes for macOS ARM64 compilation issues:
- Detects and replaces `apple-m1` with `armv8-a`
- Automatically retries compilation with corrected flags
- Uses GCC 12 for optimal compatibility

## Resources

The server provides access to:
- Project configurations (XML files)
- Simulation outputs (XML, SVG files)
- Project templates and examples
- Real-time compilation and execution

## Current Limitations

- Advanced visualization features require additional matplotlib integration
- Cell definition modification uses simplified XML handling
- Substrate analysis requires extended implementation

## Future Enhancements

- Full matplotlib integration for advanced visualizations
- Real-time simulation monitoring with progress updates
- Advanced cell definition editing with validation
- Substrate and microenvironment analysis tools
- Parameter sweep capabilities
- Integration with PhysiCell Studio
- Package distribution via PyPI

## License

MIT License