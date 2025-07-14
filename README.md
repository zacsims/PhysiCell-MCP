# PhysiCell MCP Server

An MCP (Model Context Protocol) server that provides tools to interact with PhysiCell simulations, including creating projects, configuring parameters, running simulations, and analyzing results.

## Features

The PhysiCell MCP Server provides the following tools:

- **create_project**: Create new PhysiCell projects from templates
- **configure_simulation**: Modify simulation parameters (time, domain, etc.)
- **add_cell_definition**: Add or modify cell type definitions
- **run_simulation**: Compile and run PhysiCell simulations
- **get_simulation_status**: Check the current status of a simulation
- **analyze_results**: Analyze simulation outputs (cell counts, positions, velocities)
- **visualize_snapshot**: Generate visualizations of simulation states

## Prerequisites

1. **PhysiCell**: Install PhysiCell from https://github.com/MathCancer/PhysiCell
2. **Python 3.8+**: Required for the MCP server
3. **uv**: Fast Python package manager (install from https://github.com/astral-sh/uv)

## Installation

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or on Windows:
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone or download the PhysiCell MCP server files

3. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

4. Ensure PhysiCell is installed and compiled

## Configuration

### For Claude Desktop

Add the following to your Claude Desktop configuration file:

**On macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**On Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "physicell": {
      "command": "uv",
      "args": ["run", "python", "/path/to/physicell_mcp_server.py", "--physicell-path", "/path/to/PhysiCell"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Replace `/path/to/physicell_mcp_server.py` with the actual path to the server script, and `/path/to/PhysiCell` with your PhysiCell installation directory.

### Alternative: Using uvx for isolated execution

You can also run the server in an isolated environment using `uvx`:

```json
{
  "mcpServers": {
    "physicell": {
      "command": "uvx",
      "args": ["--from", "/path/to/physicell-mcp-server", "physicell-mcp-server", "--physicell-path", "/path/to/PhysiCell"],
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

"Create a new PhysiCell project called 'tumor_growth' using the biorobots template"

### Configuring a Simulation

"Configure the tumor_growth project to run for 1440 minutes with a time step of 6 minutes"

### Running a Simulation

"Run the tumor_growth simulation with 8 threads"

### Analyzing Results

"Analyze the cell count in the tumor_growth project"
"Show me the cell positions at time 720 minutes"

## Server Arguments

- `--physicell-path`: Path to your PhysiCell installation directory (default: `./PhysiCell`)

## Development

### Running Standalone

You can test the server standalone:

```bash
uv run python physicell_mcp_server.py --physicell-path /path/to/PhysiCell
```

### Installing Additional Dependencies

To add new dependencies:

```bash
uv pip install package-name
uv pip freeze > requirements.txt
```

### Adding New Tools

To add new tools, modify the `list_tools()` method to define the tool schema and implement the corresponding handler in `call_tool()`.

## Quick Start with uv

For a complete setup from scratch:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository (or download the files)
git clone <repository-url>
cd physicell-mcp-server

# Set up Python environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Test the server
uv run python physicell_mcp_server.py --help
```

## Troubleshooting

1. **PhysiCell not found**: Ensure the `--physicell-path` argument points to your PhysiCell installation
2. **Compilation errors**: Make sure you have all PhysiCell dependencies installed (OpenMP, etc.)
3. **Permission errors**: Ensure you have write permissions in the PhysiCell directory
4. **uv not found**: Make sure uv is installed and in your PATH
5. **Module not found**: Ensure you've activated the virtual environment and installed dependencies

## Resources

The server provides access to:
- Project configurations (XML files)
- Simulation outputs
- Project information

## Limitations

- Visualization currently returns placeholder text (actual implementation would use matplotlib)
- Cell definition modification is partially implemented
- Substrate analysis requires additional implementation

## Future Enhancements

- Full matplotlib integration for visualizations
- Real-time simulation monitoring
- Advanced cell definition editing
- Substrate and microenvironment analysis
- Parameter sweep capabilities
- Integration with PhysiCell Studio
- Package distribution via PyPI for easier `uvx` usage

## License

MIT License
