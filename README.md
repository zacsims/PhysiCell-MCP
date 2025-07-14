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
3. **MCP SDK**: Install with `pip install mcp`

## Installation

1. Clone or download the PhysiCell MCP server files
2. Install the MCP Python SDK:
   ```bash
   pip install mcp
   ```

3. Ensure PhysiCell is installed and compiled

## Configuration

### For Claude Desktop

Add the following to your Claude Desktop configuration file:

**On macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**On Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "physicell": {
      "command": "python",
      "args": ["/path/to/physicell_mcp_server.py", "--physicell-path", "/path/to/PhysiCell"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Replace `/path/to/physicell_mcp_server.py` with the actual path to the server script, and `/path/to/PhysiCell` with your PhysiCell installation directory.

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
python physicell_mcp_server.py --physicell-path /path/to/PhysiCell
```

### Adding New Tools

To add new tools, modify the `list_tools()` method to define the tool schema and implement the corresponding handler in `call_tool()`.

## Troubleshooting

1. **PhysiCell not found**: Ensure the `--physicell-path` argument points to your PhysiCell installation
2. **Compilation errors**: Make sure you have all PhysiCell dependencies installed (OpenMP, etc.)
3. **Permission errors**: Ensure you have write permissions in the PhysiCell directory

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

## License

MIT License
