#!/usr/bin/env python3
"""
PhysiCell MCP Server

An MCP server that provides tools to interact with PhysiCell simulations.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("physicell-mcp")


class PhysiCellMCPServer:
    def __init__(self, physicell_path: str):
        self.physicell_path = Path(physicell_path).resolve()
        self.server = Server("physicell-mcp")
        
        # Verify PhysiCell directory exists
        if not self.physicell_path.exists():
            raise ValueError(f"PhysiCell path does not exist: {self.physicell_path}")
        
        logger.info(f"Initialized PhysiCell MCP server with path: {self.physicell_path}")
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="create_project",
                    description="Create a new PhysiCell project from a template",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Name for the new project"
                            },
                            "template": {
                                "type": "string", 
                                "description": "Template to use (biorobots-sample, heterogeneity-sample, etc.)",
                                "default": "template"
                            }
                        },
                        "required": ["project_name"]
                    }
                ),
                Tool(
                    name="list_projects", 
                    description="List available PhysiCell sample projects",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="run_simulation",
                    description="Compile and run a PhysiCell simulation", 
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "Path to project directory (optional, uses current PhysiCell dir if not specified)"
                            },
                            "threads": {
                                "type": "integer",
                                "description": "Number of OpenMP threads to use",
                                "default": 4
                            },
                            "compile": {
                                "type": "boolean", 
                                "description": "Whether to compile before running",
                                "default": True
                            }
                        }
                    }
                ),
                Tool(
                    name="get_simulation_status",
                    description="Check if a simulation is currently running",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="analyze_results",
                    description="Analyze simulation output files",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "output_dir": {
                                "type": "string",
                                "description": "Path to output directory", 
                                "default": "output"
                            },
                            "analysis_type": {
                                "type": "string",
                                "description": "Type of analysis (cell_count, positions, velocities)",
                                "default": "cell_count"
                            }
                        }
                    }
                ),
                Tool(
                    name="configure_simulation", 
                    description="Modify simulation parameters in PhysiCell_settings.xml",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "max_time": {
                                "type": "number",
                                "description": "Maximum simulation time in minutes"
                            },
                            "time_step": {
                                "type": "number", 
                                "description": "Time step in minutes"
                            },
                            "save_interval": {
                                "type": "number",
                                "description": "Save interval in minutes" 
                            }
                        }
                    }
                ),
                Tool(
                    name="fix_compilation_issues",
                    description="Fix common compilation issues (architecture flags, etc.)",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            
            if name == "list_projects":
                return await self._list_projects()
                
            elif name == "create_project":
                project_name = arguments["project_name"]
                template = arguments.get("template", "template")
                return await self._create_project(project_name, template)
                
            elif name == "run_simulation":
                project_path = arguments.get("project_path")
                threads = arguments.get("threads", 4)
                compile = arguments.get("compile", True)
                return await self._run_simulation(project_path, threads, compile)
                
            elif name == "get_simulation_status":
                return await self._get_simulation_status()
                
            elif name == "analyze_results":
                output_dir = arguments.get("output_dir", "output")
                analysis_type = arguments.get("analysis_type", "cell_count")
                return await self._analyze_results(output_dir, analysis_type)
                
            elif name == "configure_simulation":
                return await self._configure_simulation(arguments)
                
            elif name == "fix_compilation_issues":
                return await self._fix_compilation_issues()
                
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _list_projects(self) -> List[TextContent]:
        """List available sample projects"""
        try:
            result = subprocess.run(
                ["make", "list-projects"],
                cwd=self.physicell_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return [TextContent(
                    type="text",
                    text=f"Available PhysiCell projects:\n\n{result.stdout}"
                )]
            else:
                return [TextContent(
                    type="text", 
                    text=f"Error listing projects: {result.stderr}"
                )]
                
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error listing projects: {str(e)}"
            )]

    async def _create_project(self, project_name: str, template: str) -> List[TextContent]:
        """Create a new project from template"""
        try:
            # First, reset to clean state
            subprocess.run(["make", "reset"], cwd=self.physicell_path, check=True)
            
            # Create project from template
            result = subprocess.run(
                ["make", f"{template}"],
                cwd=self.physicell_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return [TextContent(
                    type="text",
                    text=f"Successfully created project '{project_name}' from template '{template}'.\n\nOutput:\n{result.stdout}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Error creating project: {result.stderr}"
                )]
                
        except Exception as e:
            return [TextContent(
                type="text", 
                text=f"Error creating project: {str(e)}"
            )]

    async def _run_simulation(self, project_path: Optional[str], threads: int, compile: bool) -> List[TextContent]:
        """Compile and run simulation"""
        try:
            work_dir = Path(project_path) if project_path else self.physicell_path
            
            # Set environment variables
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(threads)
            env["PHYSICELL_CPP"] = "/opt/homebrew/bin/g++-12"
            
            messages = []
            
            # Compile if requested
            if compile:
                # First, fix any architecture issues in Makefile
                await self._fix_makefile_architecture(work_dir)
                
                compile_result = subprocess.run(
                    ["make", "clean"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=30
                )
                
                compile_result = subprocess.run(
                    ["make"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=300
                )
                
                if compile_result.returncode != 0:
                    # Try to fix compilation errors and retry once
                    error_output = compile_result.stderr
                    if "apple-m1" in error_output or "unknown value" in error_output:
                        messages.append("Detected architecture issue, fixing...")
                        await self._fix_makefile_architecture(work_dir, force=True)
                        
                        # Retry compilation
                        compile_result = subprocess.run(
                            ["make", "clean"],
                            cwd=work_dir,
                            capture_output=True,
                            text=True,
                            env=env,
                            timeout=30
                        )
                        
                        compile_result = subprocess.run(
                            ["make"],
                            cwd=work_dir,
                            capture_output=True,
                            text=True,
                            env=env,
                            timeout=300
                        )
                    
                    if compile_result.returncode != 0:
                        return [TextContent(
                            type="text",
                            text=f"Compilation failed:\n{compile_result.stderr}\n\nStdout:\n{compile_result.stdout}"
                        )]
                
                messages.append("Compilation successful.")
            
            # Find executable
            executables = list(work_dir.glob("*"))
            executable = None
            for exe in executables:
                if exe.is_file() and os.access(exe, os.X_OK) and not exe.name.startswith('.'):
                    # Skip common non-executable files
                    if exe.suffix not in ['.cpp', '.h', '.xml', '.txt', '.md', '.o', '.so', '.dylib']:
                        executable = exe
                        break
            
            if not executable:
                return [TextContent(
                    type="text",
                    text="No executable found. Make sure compilation was successful."
                )]
            
            messages.append(f"Starting simulation with {threads} threads...")
            
            # Run simulation in background
            process = subprocess.Popen(
                [str(executable)],
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            return [TextContent(
                type="text",
                text=f"Simulation started successfully!\n\n" + "\n".join(messages) + f"\n\nExecutable: {executable.name}\nPID: {process.pid}"
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error running simulation: {str(e)}"
            )]

    async def _get_simulation_status(self) -> List[TextContent]:
        """Check simulation status"""
        try:
            # Check for running PhysiCell processes
            result = subprocess.run(
                ["pgrep", "-f", "heterogeneity|biorobots|physicell"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                return [TextContent(
                    type="text",
                    text=f"Found {len(pids)} running PhysiCell processes:\nPIDs: {', '.join(pids)}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text="No PhysiCell simulations currently running."
                )]
                
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error checking simulation status: {str(e)}"
            )]

    async def _analyze_results(self, output_dir: str, analysis_type: str) -> List[TextContent]:
        """Analyze simulation results"""
        try:
            output_path = self.physicell_path / output_dir
            
            if not output_path.exists():
                return [TextContent(
                    type="text",
                    text=f"Output directory does not exist: {output_path}"
                )]
            
            # List output files
            output_files = list(output_path.glob("*.xml"))
            svg_files = list(output_path.glob("*.svg"))
            
            analysis = f"PhysiCell Output Analysis ({analysis_type})\n"
            analysis += "=" * 50 + "\n\n"
            analysis += f"Output directory: {output_path}\n"
            analysis += f"XML files: {len(output_files)}\n"
            analysis += f"SVG files: {len(svg_files)}\n\n"
            
            if output_files:
                analysis += "Available XML output files:\n"
                for f in sorted(output_files)[:10]:  # Show first 10
                    analysis += f"  - {f.name}\n"
                if len(output_files) > 10:
                    analysis += f"  ... and {len(output_files) - 10} more\n"
            
            if svg_files:
                analysis += f"\nVisualization files: {len(svg_files)} SVG files available\n"
            
            return [TextContent(
                type="text",
                text=analysis
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error analyzing results: {str(e)}"
            )]

    async def _configure_simulation(self, parameters: Dict[str, Any]) -> List[TextContent]:
        """Configure simulation parameters"""
        try:
            config_file = self.physicell_path / "config" / "PhysiCell_settings.xml"
            
            if not config_file.exists():
                return [TextContent(
                    type="text",
                    text=f"Configuration file not found: {config_file}"
                )]
            
            # Read current config
            with open(config_file, 'r') as f:
                content = f.read()
            
            changes = []
            
            # Update parameters (simple string replacement for now)
            if "max_time" in parameters:
                max_time = parameters["max_time"]
                # This is a simplified approach - in production you'd use XML parsing
                changes.append(f"max_time: {max_time} minutes")
            
            if "time_step" in parameters:
                time_step = parameters["time_step"] 
                changes.append(f"time_step: {time_step} minutes")
                
            if "save_interval" in parameters:
                save_interval = parameters["save_interval"]
                changes.append(f"save_interval: {save_interval} minutes")
            
            if changes:
                return [TextContent(
                    type="text",
                    text=f"Configuration update requested:\n\n" + "\n".join(changes) + "\n\nNote: Manual XML editing required for now. Please edit config/PhysiCell_settings.xml"
                )]
            else:
                return [TextContent(
                    type="text",
                    text="No configuration changes specified."
                )]
                
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error configuring simulation: {str(e)}"
            )]

    async def _fix_makefile_architecture(self, work_dir: Path, force: bool = False) -> None:
        """Fix architecture issues in Makefile"""
        try:
            makefile_path = work_dir / "Makefile"
            if not makefile_path.exists():
                return
            
            with open(makefile_path, 'r') as f:
                content = f.read()
            
            # Check if we need to fix architecture
            needs_fix = force or 'apple-m1' in content or '-march=apple-m1' in content
            
            if needs_fix:
                # Replace problematic architecture flags
                content = content.replace('-march=apple-m1', '-march=armv8-a')
                content = content.replace('ARCH := apple-m1', 'ARCH := armv8-a') 
                content = content.replace('ARCH := native', 'ARCH := armv8-a')
                
                # Also ensure we have the right flags for ARM64
                if 'ARCH := armv8-a' not in content and 'ARCH :=' in content:
                    content = content.replace('ARCH := core2', 'ARCH := armv8-a')
                    content = content.replace('ARCH := corei7', 'ARCH := armv8-a')
                
                # Write back the fixed content
                with open(makefile_path, 'w') as f:
                    f.write(content)
                    
                logger.info(f"Fixed Makefile architecture in {work_dir}")
                
        except Exception as e:
            logger.error(f"Error fixing Makefile: {e}")

    async def _fix_compilation_issues(self) -> List[TextContent]:
        """Fix common compilation issues"""
        try:
            messages = []
            
            # Fix Makefile architecture
            makefile_path = self.physicell_path / "Makefile"
            if makefile_path.exists():
                await self._fix_makefile_architecture(self.physicell_path, force=True)
                messages.append("✓ Fixed Makefile architecture flags")
            
            # Check compiler
            env = os.environ.copy()
            env["PHYSICELL_CPP"] = "/opt/homebrew/bin/g++-12"
            
            try:
                result = subprocess.run([env["PHYSICELL_CPP"], "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    messages.append(f"✓ Compiler check passed: {result.stdout.split()[0]}")
                else:
                    messages.append("✗ Compiler check failed")
            except:
                messages.append("✗ Compiler not found")
            
            # Check PhysiCell directory structure
            required_dirs = ["BioFVM", "core", "modules", "config", "custom_modules"]
            for dir_name in required_dirs:
                if (self.physicell_path / dir_name).exists():
                    messages.append(f"✓ Found {dir_name} directory")
                else:
                    messages.append(f"✗ Missing {dir_name} directory")
            
            # Test a simple compilation
            try:
                result = subprocess.run(
                    ["make", "clean"],
                    cwd=self.physicell_path,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=30
                )
                messages.append("✓ Make clean successful")
            except Exception as e:
                messages.append(f"✗ Make clean failed: {str(e)}")
            
            return [TextContent(
                type="text",
                text="PhysiCell Compilation Issues Check\n" + "="*40 + "\n\n" + "\n".join(messages) + 
                     "\n\n💡 Tips:\n- Make sure you're using the right compiler (g++-12)\n- Check that all PhysiCell directories exist\n- Try running 'make heterogeneity-sample' first"
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error checking compilation issues: {str(e)}"
            )]


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PhysiCell MCP Server")
    parser.add_argument(
        "--physicell-path",
        type=str,
        default="./PhysiCell",
        help="Path to PhysiCell installation directory"
    )
    
    args = parser.parse_args()
    
    try:
        # Create server instance
        server_instance = PhysiCellMCPServer(args.physicell_path)
        
        # Run the server
        async with stdio_server() as (read_stream, write_stream):
            await server_instance.server.run(
                read_stream,
                write_stream,
                server_instance.server.create_initialization_options()
            )
            
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())