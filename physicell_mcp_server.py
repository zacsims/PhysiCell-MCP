#!/usr/bin/env python3
"""
PhysiCell MCP Server

An MCP server that provides tools to interact with PhysiCell simulations.
"""

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("physicell-mcp")


class PhysiCellMCPServer:
    def __init__(self, physicell_path: str):
        # If path starts with ./, make it relative to the script directory
        if physicell_path.startswith("./"):
            script_dir = Path(__file__).parent
            self.physicell_path = (script_dir / physicell_path[2:]).resolve()
        else:
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
                    description="Create a new PhysiCell project with proper directory structure",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Name for the new project"
                            },
                            "project_path": {
                                "type": "string",
                                "description": "Path where to create the project (default: current directory)"
                            },
                            "cell_types": {
                                "type": "array",
                                "description": "List of cell types to include",
                                "items": {"type": "string"},
                                "default": ["tumor", "macrophage", "CD8 T cell"]
                            },
                            "domain_size": {
                                "type": "number",
                                "description": "Domain size in microns (symmetric x/y/z)",
                                "default": 750
                            },
                            "simulation_time": {
                                "type": "number",
                                "description": "Total simulation time in minutes",
                                "default": 7200
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
                            "project_path": {
                                "type": "string",
                                "description": "Path to the project directory (optional)"
                            },
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
                ),
                Tool(
                    name="create_rules",
                    description="Create or modify cell behavior rules in grammar format",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "Path to the project directory"
                            },
                            "rules": {
                                "type": "array",
                                "description": "List of rule dictionaries",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "cell_type": {"type": "string"},
                                        "input": {"type": "string"},
                                        "input_change": {"type": "string", "enum": ["increases", "decreases"]},
                                        "output": {"type": "string"},
                                        "threshold": {"type": "number", "default": 0},
                                        "half_max": {"type": "number", "default": 0.5},
                                        "hill_coefficient": {"type": "number", "default": 4},
                                        "output_state": {"type": "integer", "default": 0}
                                    },
                                    "required": ["cell_type", "input", "input_change", "output"]
                                }
                            },
                            "append": {
                                "type": "boolean",
                                "description": "Append to existing rules (true) or replace (false)",
                                "default": False
                            }
                        },
                        "required": ["project_path", "rules"]
                    }
                ),
                Tool(
                    name="initialize_cells",
                    description="Initialize cell positions for simulation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "Path to the project directory"
                            },
                            "cells": {
                                "type": "array",
                                "description": "List of cell initialization configs",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "cell_type": {"type": "string"},
                                        "count": {"type": "integer"},
                                        "distribution": {
                                            "type": "string",
                                            "enum": ["random", "sphere", "box", "single"],
                                            "default": "random"
                                        },
                                        "center": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "default": [0, 0, 0]
                                        },
                                        "radius": {
                                            "type": "number",
                                            "description": "Radius for sphere distribution",
                                            "default": 100
                                        },
                                        "bounds": {
                                            "type": "array",
                                            "description": "[xmin, xmax, ymin, ymax, zmin, zmax] for box distribution",
                                            "items": {"type": "number"}
                                        }
                                    },
                                    "required": ["cell_type", "count"]
                                }
                            },
                            "append": {
                                "type": "boolean",
                                "description": "Append to existing cells (true) or replace (false)",
                                "default": False
                            }
                        },
                        "required": ["project_path", "cells"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            
            if name == "list_projects":
                return await self._list_projects()
                
            elif name == "create_project":
                return await self._create_project(**arguments)
                
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
                
            elif name == "create_rules":
                return await self._create_rules(**arguments)
                
            elif name == "initialize_cells":
                return await self._initialize_cells(**arguments)
                
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

    async def _create_project(self, project_name: str, project_path: str = None, 
                            cell_types: List[str] = None, domain_size: float = 750,
                            simulation_time: float = 7200) -> List[TextContent]:
        """Create a new PhysiCell project with proper directory structure"""
        try:
            # Set defaults
            if cell_types is None:
                cell_types = ["tumor", "macrophage", "CD8 T cell"]
            
            # Determine project directory
            if project_path:
                base_path = Path(project_path)
            else:
                base_path = Path.cwd()
            
            project_dir = base_path / project_name
            config_dir = project_dir / "config"
            
            # Create directories
            project_dir.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(exist_ok=True)
            custom_modules_dir = project_dir / "custom_modules"
            custom_modules_dir.mkdir(exist_ok=True)
            
            # Create placeholder file in custom_modules to prevent compilation errors
            placeholder_path = custom_modules_dir / ".gitkeep"
            with open(placeholder_path, 'w') as f:
                f.write("# Placeholder file for custom modules directory\n")
            
            # Generate configuration files
            settings_xml = self._generate_settings_xml(cell_types, domain_size, simulation_time)
            settings_path = config_dir / "PhysiCell_settings.xml"
            with open(settings_path, 'w') as f:
                f.write(settings_xml)
            
            # Generate default cell rules
            rules_csv = self._generate_default_rules(cell_types)
            rules_path = config_dir / "cell_rules.csv"
            with open(rules_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rules_csv)
            
            # Generate initial cell positions
            cells_csv = self._generate_initial_cells(cell_types, domain_size)
            cells_path = config_dir / "cells.csv"
            with open(cells_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(cells_csv)
            
            # Create basic main.cpp if needed
            main_cpp = self._generate_main_cpp(cell_types)
            main_path = project_dir / "main.cpp"
            with open(main_path, 'w') as f:
                f.write(main_cpp)
            
            # Create Makefile
            makefile = self._generate_makefile(project_name)
            makefile_path = project_dir / "Makefile"
            with open(makefile_path, 'w') as f:
                f.write(makefile)
            
            return [TextContent(
                type="text",
                text=f"""Successfully created PhysiCell project '{project_name}' at {project_dir}

Project structure:
{project_name}/
├── config/
│   ├── PhysiCell_settings.xml
│   ├── cell_rules.csv ({len(rules_csv)-1} rules)
│   └── cells.csv ({len(cells_csv)-1} initial cells)
├── custom_modules/
├── main.cpp
└── Makefile

Cell types configured: {', '.join(cell_types)}
Domain size: ±{domain_size} µm
Simulation time: {simulation_time} minutes"""
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating project: {str(e)}"
            )]

    async def _run_simulation(self, project_path: Optional[str], threads: int, compile: bool) -> List[TextContent]:
        """Compile and run simulation"""
        try:
            if project_path:
                # If project_path starts with ./ make it relative to the script directory (where MCP server runs)
                if project_path.startswith("./"):
                    script_dir = Path(__file__).parent
                    work_dir = (script_dir / project_path[2:]).resolve()
                else:
                    work_dir = Path(project_path).resolve()
            else:
                work_dir = self.physicell_path
            
            # Check if this is a new-style project with config directory
            config_file = work_dir / "config" / "PhysiCell_settings.xml"
            if not config_file.exists():
                # Try legacy location
                config_file = work_dir / "PhysiCell_settings.xml"
                if not config_file.exists():
                    return [TextContent(
                        type="text",
                        text=f"No PhysiCell_settings.xml found in {work_dir}/config/ or {work_dir}/. Please create a project first."
                    )]
            
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
            
            # Find executable - should match the project directory name
            project_name = work_dir.name
            executable = work_dir / project_name
            
            if not executable.exists() or not executable.is_file():
                # Try some common executable names
                possible_names = [project_name, "main", "project", "simulation"]
                executable = None
                for name in possible_names:
                    candidate = work_dir / name
                    if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
                        executable = candidate
                        break
                
                if not executable:
                    return [TextContent(
                        type="text",
                        text=f"No executable found. Expected '{project_name}' in {work_dir}. Make sure compilation was successful."
                    )]
            
            if not os.access(executable, os.X_OK):
                return [TextContent(
                    type="text",
                    text=f"Executable '{executable.name}' is not executable. Check file permissions."
                )]
            
            messages.append(f"Starting simulation with {threads} threads...")
            messages.append(f"Using config: {config_file}")
            messages.append(f"Working directory: {work_dir}")
            messages.append(f"Executable: {executable.name}")
            
            # Run simulation and wait for completion
            try:
                # Use just the executable name since we're setting cwd to work_dir
                executable_name = executable.name
                process = subprocess.Popen(
                    [f"./{executable_name}"],
                    cwd=work_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for process to complete and capture output
                stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
                
                if process.returncode == 0:
                    return [TextContent(
                        type="text",
                        text=f"Simulation completed successfully!\n\n" + "\n".join(messages) + f"\n\nSimulation Output:\n{stdout}" + (f"\n\nErrors/Warnings:\n{stderr}" if stderr.strip() else "")
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Simulation failed with exit code {process.returncode}\n\n" + "\n".join(messages) + f"\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}"
                    )]
                    
            except subprocess.TimeoutExpired:
                process.kill()
                return [TextContent(
                    type="text",
                    text=f"Simulation timed out after 5 minutes and was terminated.\n\n" + "\n".join(messages)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error running simulation: {str(e)}\n\nExecutable: {executable}\nWorking directory: {work_dir}"
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
            # Look for config file in multiple locations
            config_file = None
            if "project_path" in parameters:
                project_path = Path(parameters["project_path"])
                config_file = project_path / "config" / "PhysiCell_settings.xml"
                if not config_file.exists():
                    config_file = project_path / "PhysiCell_settings.xml"
            else:
                config_file = self.physicell_path / "config" / "PhysiCell_settings.xml"
                if not config_file.exists():
                    config_file = self.physicell_path / "PhysiCell_settings.xml"
            
            if not config_file or not config_file.exists():
                return [TextContent(
                    type="text",
                    text=f"Configuration file not found. Please specify project_path or create a project first."
                )]
            
            # Parse XML
            tree = ET.parse(config_file)
            root = tree.getroot()
            
            changes = []
            
            # Update parameters using proper XML parsing
            if "max_time" in parameters:
                max_time = parameters["max_time"]
                overall = root.find('overall')
                if overall is not None:
                    max_time_elem = overall.find('max_time')
                    if max_time_elem is not None:
                        max_time_elem.text = str(max_time)
                        changes.append(f"max_time: {max_time} minutes")
            
            if "time_step" in parameters:
                time_step = parameters["time_step"]
                overall = root.find('overall')
                if overall is not None:
                    # Update diffusion time step
                    dt_diffusion = overall.find('dt_diffusion')
                    if dt_diffusion is not None:
                        dt_diffusion.text = str(time_step)
                        changes.append(f"dt_diffusion: {time_step} minutes")
                
            if "save_interval" in parameters:
                save_interval = parameters["save_interval"]
                save = root.find('save')
                if save is not None:
                    full_data = save.find('full_data')
                    if full_data is not None:
                        interval = full_data.find('interval')
                        if interval is not None:
                            interval.text = str(save_interval)
                            changes.append(f"save_interval: {save_interval} minutes")
                    
                    svg = save.find('SVG')
                    if svg is not None:
                        interval = svg.find('interval')
                        if interval is not None:
                            interval.text = str(save_interval)
                            changes.append(f"SVG save_interval: {save_interval} minutes")
            
            if changes:
                # Write updated XML back to file
                tree.write(config_file, encoding='UTF-8', xml_declaration=True)
                return [TextContent(
                    type="text",
                    text=f"Successfully updated configuration at {config_file}:\n\n" + "\n".join(changes)
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
        """Fix architecture and compiler issues in Makefile for macOS"""
        try:
            makefile_path = work_dir / "Makefile"
            if not makefile_path.exists():
                return
            
            with open(makefile_path, 'r') as f:
                content = f.read()
            
            # Check if we need to fix architecture or compiler
            needs_fix = (force or 'apple-m1' in content or '-march=apple-m1' in content or 
                        'CC := g++' in content or 'ARCH := native' in content)
            
            if needs_fix:
                # Replace problematic architecture flags
                content = content.replace('-march=apple-m1', '-march=armv8-a')
                content = content.replace('ARCH := apple-m1', 'ARCH := armv8-a') 
                content = content.replace('ARCH := native', 'ARCH := armv8-a')
                content = content.replace('ARCH := core2', 'ARCH := armv8-a')
                content = content.replace('ARCH := corei7', 'ARCH := armv8-a')
                
                # Fix compiler to use Homebrew g++ with OpenMP support
                content = content.replace('CC := g++', 'CC := /opt/homebrew/bin/g++-12')
                content = content.replace('CC := clang++', 'CC := /opt/homebrew/bin/g++-12')
                
                # Fix custom modules to be optional  
                if 'custom_modules/*.cpp' in content and 'wildcard' not in content:
                    content = content.replace('custom_modules/*.cpp', '$(wildcard custom_modules/*.cpp)')
                
                # Replace wildcard compilation with selective compilation to avoid duplicate symbols
                if '$(PHYSICELL_DIR)/BioFVM/*.cpp' in content:
                    content = content.replace('$(PHYSICELL_DIR)/BioFVM/*.cpp', '$(BIOFVM_SOURCES)')
                if '$(PHYSICELL_DIR)/core/*.cpp' in content:
                    content = content.replace('$(PHYSICELL_DIR)/core/*.cpp', '$(CORE_SOURCES)')
                if '$(PHYSICELL_DIR)/modules/*.cpp' in content:
                    content = content.replace('$(PHYSICELL_DIR)/modules/*.cpp', '$(MODULE_SOURCES)')
                
                # Add source definitions if not present
                if 'BIOFVM_SOURCES' not in content:
                    makefile_vars = '''
# Source files - be selective to avoid duplicate symbols
BIOFVM_SOURCES := $(PHYSICELL_DIR)/BioFVM/BioFVM_vector.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_agent_container.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_mesh.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_microenvironment.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_solvers.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_utilities.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_basic_agent.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_matlab.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_MultiCellDS.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/pugixml.cpp

CORE_SOURCES := $(PHYSICELL_DIR)/core/PhysiCell_phenotype.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_cell_container.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_standard_models.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_cell.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_custom.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_utilities.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_constants.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_basic_signaling.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_rules.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_signal_behavior.cpp

MODULE_SOURCES := $(PHYSICELL_DIR)/modules/PhysiCell_SVG.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_pathology.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_MultiCellDS.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_various_outputs.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_pugixml.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_settings.cpp

'''
                    # Insert after INCLUDES line
                    content = content.replace('INCLUDES := $(BioFVM_INCLUDES) $(PhysiCell_INCLUDES)', 
                                            'INCLUDES := $(BioFVM_INCLUDES) $(PhysiCell_INCLUDES)' + makefile_vars)
                
                # Suppress warnings with -w flag
                if '-w' not in content and 'CFLAGS' in content:
                    content = content.replace('-std=c++11', '-std=c++11 -w')
                
                # Write back the fixed content
                with open(makefile_path, 'w') as f:
                    f.write(content)
                    
                logger.info(f"Fixed Makefile architecture and compiler in {work_dir}")
                
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

    async def _create_rules(self, project_path: str, rules: List[Dict], append: bool = False) -> List[TextContent]:
        """Create or modify cell behavior rules"""
        try:
            config_dir = Path(project_path) / "config"
            rules_file = config_dir / "cell_rules.csv"
            
            # Ensure config directory exists
            if not config_dir.exists():
                return [TextContent(
                    type="text",
                    text=f"Config directory not found at {config_dir}. Please create a project first."
                )]
            
            # Prepare rule data
            headers = ["cell type", "input", "input change", "output", "threshold", "half-max", "hill coefficient", "output state"]
            rows = []
            
            # If appending, read existing rules
            if append and rules_file.exists():
                with open(rules_file, 'r') as f:
                    reader = csv.reader(f)
                    existing = list(reader)
                    if existing and existing[0] == headers:
                        rows.extend(existing[1:])  # Skip header
            
            # Add new rules
            for rule in rules:
                row = [
                    rule.get("cell_type", ""),
                    rule.get("input", ""),
                    rule.get("input_change", ""),
                    rule.get("output", ""),
                    rule.get("threshold", 0),
                    rule.get("half_max", 0.5),
                    rule.get("hill_coefficient", 4),
                    rule.get("output_state", 0)
                ]
                rows.append(row)
            
            # Write rules to file
            with open(rules_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            
            action = "appended to" if append else "created"
            return [TextContent(
                type="text",
                text=f"Successfully {action} cell rules at {rules_file}\nTotal rules: {len(rows)}"
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating rules: {str(e)}"
            )]

    async def _initialize_cells(self, project_path: str, cells: List[Dict], append: bool = False) -> List[TextContent]:
        """Initialize cell positions"""
        try:
            config_dir = Path(project_path) / "config"
            cells_file = config_dir / "cells.csv"
            
            # Ensure config directory exists
            if not config_dir.exists():
                return [TextContent(
                    type="text",
                    text=f"Config directory not found at {config_dir}. Please create a project first."
                )]
            
            # Prepare cell data
            headers = ["x", "y", "z", "type", "volume", "cycle entry", "custom:GFP", "custom:sample"]
            rows = []
            
            # If appending, read existing cells
            if append and cells_file.exists():
                with open(cells_file, 'r') as f:
                    reader = csv.reader(f)
                    existing = list(reader)
                    if existing and existing[0] == headers:
                        rows.extend(existing[1:])  # Skip header
            
            # Generate new cell positions
            for cell_config in cells:
                cell_type = cell_config["cell_type"]
                count = cell_config["count"]
                distribution = cell_config.get("distribution", "random")
                center = cell_config.get("center", [0, 0, 0])
                
                for _ in range(count):
                    if distribution == "random":
                        bounds = cell_config.get("bounds", [-500, 500, -500, 500, -50, 50])
                        x = random.uniform(bounds[0], bounds[1])
                        y = random.uniform(bounds[2], bounds[3])
                        z = random.uniform(bounds[4], bounds[5])
                    elif distribution == "sphere":
                        radius = cell_config.get("radius", 100)
                        # Random point in sphere
                        theta = random.uniform(0, 2 * 3.14159)
                        phi = random.uniform(0, 3.14159)
                        r = radius * random.random() ** (1/3)
                        x = center[0] + r * math.sin(phi) * math.cos(theta)
                        y = center[1] + r * math.sin(phi) * math.sin(theta)
                        z = center[2] + r * math.cos(phi)
                    elif distribution == "single":
                        x, y, z = center
                    else:  # box
                        bounds = cell_config.get("bounds", [-100, 100, -100, 100, -10, 10])
                        x = center[0] + random.uniform(bounds[0], bounds[1])
                        y = center[1] + random.uniform(bounds[2], bounds[3])
                        z = center[2] + random.uniform(bounds[4], bounds[5])
                    
                    row = [x, y, z, cell_type, "", "", "", ""]
                    rows.append(row)
            
            # Write cells to file
            with open(cells_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            
            action = "appended" if append else "initialized"
            return [TextContent(
                type="text",
                text=f"Successfully {action} {len(rows)} cells at {cells_file}"
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error initializing cells: {str(e)}"
            )]

    def _generate_settings_xml(self, cell_types: List[str], domain_size: float, simulation_time: float) -> str:
        """Generate a basic PhysiCell settings XML file"""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<PhysiCell_settings version="devel-version">
    <cell_rules>
        <rulesets>
            <ruleset protocol="CBHG" version="2.0" format="csv" enabled="true">
                <folder>./config</folder>
                <filename>cell_rules.csv</filename>
            </ruleset>
        </rulesets>
    </cell_rules>
    
    <domain>
        <x_min>-{domain_size}</x_min>
        <x_max>{domain_size}</x_max>
        <y_min>-{domain_size}</y_min>
        <y_max>{domain_size}</y_max>
        <z_min>-{domain_size}</z_min>
        <z_max>{domain_size}</z_max>
        <dx>20</dx>
        <dy>20</dy>
        <dz>20</dz>
        <use_2D>false</use_2D>
    </domain>
    
    <overall>
        <max_time units="min">{simulation_time}</max_time>
        <time_units>min</time_units>
        <space_units>micron</space_units>
        <dt_diffusion units="min">0.01</dt_diffusion>
        <dt_mechanics units="min">0.1</dt_mechanics>
        <dt_phenotype units="min">6</dt_phenotype>
    </overall>
    
    <parallel>
        <omp_num_threads>8</omp_num_threads>
    </parallel>
    
    <save>
        <folder>output</folder>
        <full_data>
            <interval units="min">60</interval>
            <enable>true</enable>
        </full_data>
        <SVG>
            <interval units="min">60</interval>
            <enable>true</enable>
        </SVG>
    </save>
    
    <microenvironment_setup>
        <variable name="oxygen" units="mmHg" ID="0">
            <physical_parameter_set>
                <diffusion_coefficient units="micron^2/min">100000</diffusion_coefficient>
                <decay_rate units="1/min">0.1</decay_rate>
            </physical_parameter_set>
            <initial_condition units="mmHg">38</initial_condition>
            <Dirichlet_boundary_condition units="mmHg" enabled="true">38</Dirichlet_boundary_condition>
        </variable>
    </microenvironment_setup>
    
    <cell_definitions>
        {"".join(f'''
        <cell_definition name="{cell_type}" ID="{i}">
            <phenotype>
                <cycle code="5" name="live">
                    <phase_transition_rates units="1/min">
                        <rate start_index="0" end_index="0" fixed_duration="false">0.0</rate>
                    </phase_transition_rates>
                </cycle>
                <death>
                    <model code="100" name="apoptosis">
                        <death_rate units="1/min">0</death_rate>
                    </model>
                    <model code="101" name="necrosis">
                        <death_rate units="1/min">0</death_rate>
                    </model>
                </death>
                <volume>
                    <total units="micron^3">2494</total>
                    <fluid_fraction units="dimensionless">0.75</fluid_fraction>
                    <nuclear units="micron^3">540</nuclear>
                </volume>
                <mechanics>
                    <cell_cell_adhesion_strength units="micron/min">0.4</cell_cell_adhesion_strength>
                    <cell_cell_repulsion_strength units="micron/min">10</cell_cell_repulsion_strength>
                </mechanics>
                <motility>
                    <speed units="micron/min">1</speed>
                    <persistence_time units="min">1</persistence_time>
                    <migration_bias units="dimensionless">0.5</migration_bias>
                </motility>
            </phenotype>
        </cell_definition>''' for i, cell_type in enumerate(cell_types))}
    </cell_definitions>
    
    <initial_conditions>
        <cell_positions type="csv" enabled="true">
            <folder>./config</folder>
            <filename>cells.csv</filename>
        </cell_positions>
    </initial_conditions>
</PhysiCell_settings>"""

    def _generate_default_rules(self, cell_types: List[str]) -> List[List]:
        """Generate default cell behavior rules"""
        rules = [
            ["cell type", "input", "input change", "output", "threshold", "half-max", "hill coefficient", "output state"]
        ]
        
        # Add some basic rules for each cell type
        if "tumor" in cell_types:
            rules.extend([
                ["tumor", "oxygen", "increases", "cycle entry", 7.2e-03, 21.5, 4, 0],
                ["tumor", "pressure", "decreases", "cycle entry", 0, 0.25, 3, 0],
                ["tumor", "oxygen", "decreases", "necrosis", 0, 3.75, 8, 0],
            ])
        
        if "macrophage" in cell_types:
            rules.extend([
                ["macrophage", "oxygen", "increases", "pro-inflammatory factor secretion", 10, 12, 16, 0],
                ["macrophage", "oxygen", "decreases", "anti-inflammatory factor secretion", 0, 12, 16, 0],
            ])
        
        if "CD8 T cell" in cell_types:
            rules.extend([
                ["CD8 T cell", "contact with tumor", "decreases", "migration speed", 0, 0.5, 8, 0],
            ])
        
        return rules

    def _generate_initial_cells(self, cell_types: List[str], domain_size: float) -> List[List]:
        """Generate initial cell positions"""
        cells = [
            ["x", "y", "z", "type", "volume", "cycle entry", "custom:GFP", "custom:sample"]
        ]
        
        # Add some cells for each type
        for cell_type in cell_types:
            # Number of cells depends on type
            if cell_type == "tumor":
                count = 100
                radius = 200
            elif cell_type == "macrophage":
                count = 50
                radius = domain_size * 0.8
            else:
                count = 25
                radius = domain_size * 0.8
            
            for _ in range(count):
                if cell_type == "tumor":
                    # Cluster tumor cells near center
                    r = radius * random.random()
                    theta = random.uniform(0, 2 * 3.14159)
                    x = r * math.cos(theta)
                    y = r * math.sin(theta)
                    z = 0
                else:
                    # Distribute other cells more widely
                    x = random.uniform(-radius, radius)
                    y = random.uniform(-radius, radius)
                    z = 0
                
                cells.append([x, y, z, cell_type, "", "", "", ""])
        
        return cells

    def _generate_main_cpp(self, cell_types: List[str]) -> str:
        """Generate a complete PhysiCell simulation main.cpp file"""
        
        # Generate cell creation code for each type
        cell_creation_code = ""
        for i, cell_type in enumerate(cell_types):
            var_name = cell_type.replace(' ', '_').replace('-', '_').lower()
            count = 20 if i == 0 else 10
            positioning = '''double r = 100.0 * sqrt(UniformRandom());
                    double theta = 6.28318530718 * UniformRandom();
                    position[0] = r * cos(theta);
                    position[1] = r * sin(theta);''' if i == 0 else '''position[0] = -400 + 800 * UniformRandom();
                    position[1] = -400 + 800 * UniformRandom();'''
            
            cell_creation_code += f'''
    
    // Create {cell_type} cells
    Cell_Definition* {var_name}_def = find_cell_definition("{cell_type}");
    if ({var_name}_def != NULL)
    {{
        int {var_name}_count = 0;
        int target_{var_name}_count = {count};  // {"Primary" if i == 0 else "Secondary"} cell type
        
        for (int n = 0; n < target_{var_name}_count; n++)
        {{
            try {{
                std::vector<double> position = {{0, 0, 0}};
                
                // {"Cluster" if i == 0 else "Random"} positioning for {cell_type}
                if (n < target_{var_name}_count)
                {{
                    {positioning}
                }}
                
                pC = create_cell(*{var_name}_def);
                if (pC != NULL)
                {{
                    pC->assign_position(position);
                    {var_name}_count++;
                    total_count++;
                }}
            }}
            catch (const std::exception& e) {{
                std::cout << "Warning: Could not create {cell_type} cell " << n << ": " << e.what() << std::endl;
            }}
        }}
        
        std::cout << "Created " << {var_name}_count << " {cell_type} cells." << std::endl;
    }}
    else
    {{
        std::cout << "Warning: {cell_type} cell definition not found!" << std::endl;
    }}'''
        
        return f"""#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <fstream>

#include "../PhysiCell/core/PhysiCell.h"
#include "../PhysiCell/modules/PhysiCell_standard_modules.h"

using namespace BioFVM;
using namespace PhysiCell;

void setup_microenvironment(void);
void setup_tissue(void);

int main(int argc, char* argv[])
{{
    // load and parse settings file(s)
    bool XML_status = false;
    char copy_command[1024];
    if (argc > 1)
    {{
        XML_status = load_PhysiCell_config_file(argv[1]);
        sprintf(copy_command, "cp %s %s", argv[1], PhysiCell_settings.folder.c_str());
    }}
    else
    {{
        XML_status = load_PhysiCell_config_file("./config/PhysiCell_settings.xml");
        sprintf(copy_command, "cp ./config/PhysiCell_settings.xml %s", PhysiCell_settings.folder.c_str());
    }}
    if (!XML_status)
    {{ 
        std::cout << "Error: Could not load settings file." << std::endl;
        exit(-1); 
    }}

    // copy config file to output folder
    system(copy_command);
    
    // ensure output directory exists
    system("mkdir -p output");

    // time setup
    std::string time_units = "min";

    // Microenvironment setup
    setup_microenvironment();

    // PhysiCell setup
    
    // Initialize cell definitions from the config file
    initialize_default_cell_definition();
    
    std::cout << "Cell definitions loaded: " << cell_definitions_by_index.size() << std::endl;
    for (int i = 0; i < cell_definitions_by_index.size(); i++)
    {{
        std::cout << "  - " << cell_definitions_by_index[i]->name << " (ID: " << cell_definitions_by_index[i]->type << ")" << std::endl;
    }}
    
    // set mechanics voxel size, and match the data structure to BioFVM
    double mechanics_voxel_size = 30;
    Cell_Container* cell_container = create_cell_container_for_microenvironment(microenvironment, mechanics_voxel_size);

    // create some cells from the config file
    setup_tissue();

    // set MultiCellDS save options (some functions may not be available in all PhysiCell versions)
    // set_save_biofvm_mesh_as_matlab(true);
    // set_save_biofvm_data_as_matlab(true);

    // save a simulation snapshot
    char filename[1024];
    sprintf(filename, "%s/initial", PhysiCell_settings.folder.c_str());
    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);
    
    // SVG plotting requires additional setup - skipping for this example

    display_citations();

    // set the performance timers
    BioFVM::RUNTIME_TIC();
    BioFVM::TIC();

    std::cout << "Starting PhysiCell simulation..." << std::endl;
    std::cout << "Max time: " << PhysiCell_settings.max_time << " minutes" << std::endl;
    std::cout << "Cells initialized and ready to simulate." << std::endl;

    // main loop
    try
    {{
        while (PhysiCell_globals.current_time < PhysiCell_settings.max_time + 0.1 * diffusion_dt)
        {{
            // save data if it's time
            if (PhysiCell_globals.current_time > PhysiCell_globals.next_full_save_time - 0.5 * diffusion_dt)
            {{
                display_simulation_status(std::cout);

                // save the simulation snapshot
                sprintf(filename, "%s/output%08u", PhysiCell_settings.folder.c_str(), PhysiCell_globals.full_output_index);
                save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);

                // SVG plotting skipped for this example

                PhysiCell_globals.full_output_index++;
                PhysiCell_globals.next_full_save_time += PhysiCell_settings.full_save_interval;
            }}

            // integrate the microenvironment
            microenvironment.simulate_diffusion_decay(diffusion_dt);

            // run PhysiCell
            ((Cell_Container*)microenvironment.agent_container)->update_all_cells(PhysiCell_globals.current_time);

            PhysiCell_globals.current_time += diffusion_dt;
        }}

        // save a final simulation snapshot
        sprintf(filename, "%s/final", PhysiCell_settings.folder.c_str());
        save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);

        // Final SVG plotting skipped for this example

        // timer
        std::cout << std::endl << "Total simulation runtime: " << std::endl;
        BioFVM::display_stopwatch_value(std::cout, BioFVM::runtime_stopwatch_value());

        std::cout << std::endl << "PhysiCell simulation completed successfully!" << std::endl;
        return 0;
    }}
    catch (const std::exception& e)
    {{
        std::cout << "Error during simulation: " << e.what() << std::endl;
        return -1;
    }}
}}

void setup_microenvironment(void)
{{
    // make sure to override and go back to 2D 
    if (default_microenvironment_options.simulate_2D == false)
    {{
        std::cout << "Warning: overriding XML config option and setting to 2D!" << std::endl;
        default_microenvironment_options.simulate_2D = true;
    }}

    // initialize BioFVM
    initialize_microenvironment();

    return;
}}

void setup_tissue(void)
{{
    // Systematically create cells using the defined cell types
    
    Cell* pC;
    std::cout << "Setting up tissue..." << std::endl;
    std::cout << "Available cell definitions: " << cell_definitions_by_index.size() << std::endl;
    
    if (cell_definitions_by_index.size() == 0)
    {{
        std::cout << "Error: No cell definitions available!" << std::endl;
        return;
    }}
    
    // List cell types that were generated
    std::cout << "Cell types available: ";
    for (int i = 0; i < cell_definitions_by_index.size(); i++)
    {{
        std::cout << cell_definitions_by_index[i]->name << " (ID:" << i << ") ";
    }}
    std::cout << std::endl;
    
    // Create cells for each defined type systematically
    int total_count = 0;{cell_creation_code}
    
    std::cout << "Total cells created: " << total_count << std::endl;
    std::cout << "PhysiCell reports: " << (*all_cells).size() << " cells." << std::endl;
    
    return;
}}
"""

    def _generate_makefile(self, project_name: str) -> str:
        """Generate a Makefile for the project"""
        return f"""# Makefile for {project_name}
PROGRAM_NAME := {project_name}

# Use Homebrew g++ for OpenMP support on macOS
CC := /opt/homebrew/bin/g++-12
ARCH := armv8-a
CFLAGS := -march=$(ARCH) -O2 -fomit-frame-pointer -fopenmp -m64 -std=c++11 -w

PHYSICELL_DIR := ../PhysiCell
BioFVM_INCLUDES := -I$(PHYSICELL_DIR)/BioFVM
PhysiCell_INCLUDES := -I$(PHYSICELL_DIR)/core -I$(PHYSICELL_DIR)/modules
INCLUDES := $(BioFVM_INCLUDES) $(PhysiCell_INCLUDES)

# Source files - be selective to avoid duplicate symbols
BIOFVM_SOURCES := $(PHYSICELL_DIR)/BioFVM/BioFVM_vector.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_agent_container.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_mesh.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_microenvironment.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_solvers.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_utilities.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_basic_agent.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_matlab.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/BioFVM_MultiCellDS.cpp \\
                  $(PHYSICELL_DIR)/BioFVM/pugixml.cpp

CORE_SOURCES := $(PHYSICELL_DIR)/core/PhysiCell_phenotype.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_cell_container.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_standard_models.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_cell.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_custom.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_utilities.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_constants.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_basic_signaling.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_rules.cpp \\
                $(PHYSICELL_DIR)/core/PhysiCell_signal_behavior.cpp

MODULE_SOURCES := $(PHYSICELL_DIR)/modules/PhysiCell_SVG.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_pathology.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_MultiCellDS.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_various_outputs.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_pugixml.cpp \\
                  $(PHYSICELL_DIR)/modules/PhysiCell_settings.cpp

# Custom modules (optional)
CUSTOM_MODULES := $(wildcard custom_modules/*.cpp)

all: $(PROGRAM_NAME)

$(PROGRAM_NAME): main.cpp
\t$(CC) $(CFLAGS) $(INCLUDES) -o $(PROGRAM_NAME) main.cpp \\
\t$(BIOFVM_SOURCES) \\
\t$(CORE_SOURCES) \\
\t$(MODULE_SOURCES) \\
\t$(CUSTOM_MODULES)

clean:
\trm -f $(PROGRAM_NAME) *.o
\trm -rf output/*

.PHONY: all clean
"""


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
        # Log startup
        logger.info(f"Starting PhysiCell MCP Server...")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"PhysiCell path argument: {args.physicell_path}")
        
        # Create server instance
        server_instance = PhysiCellMCPServer(args.physicell_path)
        
        # Run the server
        logger.info("Starting MCP stdio server...")
        async with stdio_server() as (read_stream, write_stream):
            await server_instance.server.run(
                read_stream,
                write_stream,
                server_instance.server.create_initialization_options()
            )
            
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())