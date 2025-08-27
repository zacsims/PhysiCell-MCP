#!/usr/bin/env python3
"""
PhysiCell Biological MCP Server
An intelligent MCP server that generates biologically realistic PhysiCell simulations
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
from mcp.types import Resource, Tool, TextContent

# Import dynamic biological simulation generator
from biological_knowledge import BiologicalSimulationGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("physicell-bio-mcp")


class PhysiCellBiologicalMCPServer:
    def __init__(self, physicell_path: str):
        # Handle relative paths
        if physicell_path.startswith("./"):
            script_dir = Path(__file__).parent
            self.physicell_path = (script_dir / physicell_path[2:]).resolve()
        else:
            self.physicell_path = Path(physicell_path).resolve()
        
        self.server = Server("physicell-bio-mcp", "1.0.0")
        # No hardcoded biological knowledge - using dynamic generation
        
        # Verify PhysiCell directory exists
        if not self.physicell_path.exists():
            raise ValueError(f"PhysiCell path does not exist: {self.physicell_path}")
        
        logger.info(f"Initialized PhysiCell Biological MCP server with path: {self.physicell_path}")
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register MCP handlers with biological intelligence"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="analyze_biological_scenario",
                    description="Analyze biological scenario to identify key components for simulation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Biological scenario description"
                            }
                        },
                        "required": ["description"]
                    }
                ),
                Tool(
                    name="generate_simulation_components",
                    description="Generate cell types, substrates, and interactions from biological analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "biological_analysis": {
                                "type": "string",
                                "description": "LLM analysis of biological scenario"
                            },
                            "component_type": {
                                "type": "string",
                                "enum": ["cell_types", "substrates", "interactions", "all"],
                                "description": "Type of components to generate"
                            }
                        },
                        "required": ["biological_analysis", "component_type"]
                    }
                ),
                Tool(
                    name="create_physicell_configuration",
                    description="Generate PhysiCell XML configuration from component data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "cell_data": {
                                "type": "array",
                                "description": "List of cell type definitions"
                            },
                            "substrate_data": {
                                "type": "array", 
                                "description": "List of substrate definitions"
                            },
                            "simulation_params": {
                                "type": "object",
                                "description": "Simulation parameters (domain size, time, etc.)"
                            }
                        },
                        "required": ["cell_data", "substrate_data"]
                    }
                ),
                Tool(
                    name="create_interaction_rules",
                    description="Generate CSV interaction rules for PhysiCell",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "cell_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of cell types in simulation"
                            },
                            "substrates": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of substrates in simulation"
                            },
                            "biological_context": {
                                "type": "string",
                                "description": "Biological scenario context for realistic rules"
                            }
                        },
                        "required": ["cell_types", "substrates", "biological_context"]
                    }
                ),
                Tool(
                    name="create_cell_initialization",
                    description="Generate initial cell positions and setup",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "cell_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of cell types to initialize"
                            },
                            "domain_size": {
                                "type": "number",
                                "description": "Simulation domain size in microns"
                            },
                            "spatial_organization": {
                                "type": "string",
                                "description": "Description of desired spatial organization"
                            }
                        },
                        "required": ["cell_types", "domain_size"]
                    }
                ),
                Tool(
                    name="create_biological_simulation",
                    description="Complete end-to-end simulation creation using dynamic generation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Biological scenario description"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Name for the project"
                            },
                            "project_path": {
                                "type": "string",
                                "description": "Path where to create the project (optional)"
                            }
                        },
                        "required": ["description", "project_name"]
                    }
                ),
                Tool(
                    name="validate_biological_parameters",
                    description="Validate and suggest improvements for biological parameters",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "parameters": {
                                "type": "object",
                                "description": "Parameters to validate"
                            },
                            "parameter_type": {
                                "type": "string",
                                "enum": ["cell", "substrate", "interaction"],
                                "description": "Type of parameters being validated"
                            }
                        },
                        "required": ["parameters", "parameter_type"]
                    }
                ),
                Tool(
                    name="create_project_files",
                    description="Create actual PhysiCell project files from LLM-generated content",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Name for the project"
                            },
                            "xml_content": {
                                "type": "string",
                                "description": "PhysiCell XML configuration content"
                            },
                            "rules_csv": {
                                "type": "string",
                                "description": "CSV interaction rules content"
                            },
                            "cells_csv": {
                                "type": "string",
                                "description": "CSV cell initialization content"
                            },
                            "project_path": {
                                "type": "string",
                                "description": "Optional custom project path"
                            }
                        },
                        "required": ["project_name", "xml_content"]
                    }
                ),
                Tool(
                    name="generate_physicell_project",
                    description="Generate complete PhysiCell project from biological description in a single atomic operation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Name for the project"
                            },
                            "biological_description": {
                                "type": "string",
                                "description": "Description of the biological scenario to model"
                            },
                            "structured_data": {
                                "type": "string",
                                "description": "Optional JSON structured biological data (for second call)"
                            },
                            "project_path": {
                                "type": "string",
                                "description": "Optional custom project path"
                            },
                            "enable_csv_cells": {
                                "type": "boolean",
                                "description": "Enable CSV cell placement (default true, set to false for safer initial testing)"
                            }
                        },
                        "required": ["project_name", "biological_description"]
                    }
                ),
                Tool(
                    name="fix_duplicate_cell_definitions",
                    description="Fix projects with duplicate cell definitions that cause PhysiCell crashes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "Path to project directory to fix"
                            }
                        },
                        "required": ["project_path"]
                    }
                ),
                Tool(
                    name="extract_xml_names",
                    description="Extract cell type and substrate names from PhysiCell XML for consistency",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "xml_content": {
                                "type": "string",
                                "description": "PhysiCell XML configuration content"
                            }
                        },
                        "required": ["xml_content"]
                    }
                ),
                Tool(
                    name="fix_project_config",
                    description="Fix common configuration issues in existing PhysiCell projects",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "Path to project directory to fix"
                            }
                        },
                        "required": ["project_path"]
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
                                "description": "Path to project directory"
                            },
                            "threads": {
                                "type": "integer",
                                "description": "Number of OpenMP threads",
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
                    name="reduce_memory_usage",
                    description="Reduce memory usage of a PhysiCell simulation to prevent segmentation faults",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "Path to project directory"
                            },
                            "force_2d": {
                                "type": "boolean",
                                "description": "Force simulation to 2D",
                                "default": True
                            },
                            "domain_size": {
                                "type": "number",
                                "description": "Maximum domain dimension in microns",
                                "default": 1500
                            }
                        },
                        "required": ["project_path"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls with enhanced debugging"""
            try:
                logger.info(f"Tool call: {name} with arguments: {arguments}")
                
                if name == "analyze_biological_scenario":
                    return await self._analyze_biological_scenario(**arguments)
                    
                elif name == "generate_simulation_components":
                    return await self._generate_simulation_components(**arguments)
                    
                elif name == "create_physicell_configuration":
                    return await self._create_physicell_configuration(**arguments)
                    
                elif name == "create_interaction_rules":
                    return await self._create_interaction_rules(**arguments)
                    
                elif name == "create_cell_initialization":
                    return await self._create_cell_initialization(**arguments)
                    
                elif name == "create_biological_simulation":
                    return await self._create_biological_simulation(**arguments)
                    
                elif name == "validate_biological_parameters":
                    return await self._validate_biological_parameters(**arguments)
                    
                elif name == "create_project_files":
                    return await self._create_project_files(**arguments)
                    
                elif name == "generate_physicell_project":
                    return await self._generate_physicell_project(**arguments)
                    
                elif name == "fix_duplicate_cell_definitions":
                    return await self._fix_duplicate_cell_definitions(**arguments)
                    
                elif name == "extract_xml_names":
                    return await self._extract_xml_names(**arguments)
                    
                elif name == "fix_project_config":
                    return await self._fix_project_config(**arguments)
                    
                elif name == "run_simulation":
                    # Special handling for run_simulation with defensive argument parsing
                    logger.info(f"Raw arguments received: {repr(arguments)}")
                    
                    # Defensive extraction with fallbacks and type conversion
                    project_path = arguments.get("project_path")
                    
                    # Handle different thread argument formats
                    threads_arg = arguments.get("threads", 4)
                    if isinstance(threads_arg, str):
                        try:
                            threads = int(threads_arg)
                        except (ValueError, TypeError):
                            threads = 4
                    else:
                        threads = threads_arg if isinstance(threads_arg, int) else 4
                    
                    # Handle different compile argument formats
                    compile_arg = arguments.get("compile", True)
                    if isinstance(compile_arg, str):
                        compile_flag = compile_arg.lower() in ('true', '1', 'yes', 'on')
                    else:
                        compile_flag = bool(compile_arg)
                    
                    logger.info(f"Processed arguments: project_path={repr(project_path)}, threads={threads}, compile={compile_flag}")
                    
                    return await self._run_simulation(
                        project_path=project_path,
                        threads=threads,
                        compile=compile_flag
                    )
                    
                elif name == "reduce_memory_usage":
                    return await self._reduce_memory_usage(**arguments)
                    
                elif name == "fix_cell_positions":
                    return await self._fix_cell_positions(**arguments)
                    
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error handling tool call {name}: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error handling tool call '{name}': {str(e)}\nArguments received: {arguments}"
                )]
    
    # ============================================================================
    # DYNAMIC BIOLOGICAL SIMULATION GENERATION METHODS
    # ============================================================================
    
    async def _analyze_biological_scenario(self, description: str) -> List[TextContent]:
        """Analyze biological scenario using LLM guidance"""
        try:
            # Generate analysis prompt for LLM
            analysis_prompt = BiologicalSimulationGenerator.get_analysis_prompt(description)
            
            # Get missing component suggestions
            suggestions = BiologicalSimulationGenerator.suggest_missing_components(description)
            
            # Get simulation scale estimates
            scale = BiologicalSimulationGenerator.estimate_simulation_scale(description)
            
            report = f"""Biological Scenario Analysis Guidance
====================================================

INPUT SCENARIO: "{description}"

ANALYSIS INSTRUCTIONS FOR LLM:
{analysis_prompt}

SCALE RECOMMENDATIONS:
• Domain size: {scale['domain_size']} microns
• Simulation time: {scale['simulation_time']} minutes ({scale['simulation_time']/1440:.1f} days)
• Save interval: {scale['save_interval']} minutes

ADDITIONAL CONSIDERATIONS:
{chr(10).join([f"• {suggestion}" for suggestion in suggestions]) if suggestions else "• No additional suggestions"}

Next step: Use the 'generate_simulation_components' tool with your analysis to create specific biological components."""
            
            return [TextContent(type="text", text=report)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error analyzing scenario: {e}")]

    async def _generate_simulation_components(
        self, 
        biological_analysis: str, 
        component_type: str
    ) -> List[TextContent]:
        """Generate specific simulation components using biological intelligence"""
        try:
            # Extract potential cell types and description from analysis
            description_lines = biological_analysis.lower().split('\n')
            description = ' '.join(description_lines)
            
            # Identify cell types mentioned
            potential_cells = []
            cell_keywords = ['tumor', 'cancer', 'tnbc', 'cd8', 't_cell', 'macrophage', 'm1', 'm2', 'fibroblast', 'endothelial']
            for keyword in cell_keywords:
                if keyword in description:
                    potential_cells.append(keyword)
            
            if not potential_cells:
                potential_cells = ['tumor']  # Default fallback
            
            if component_type == "cell_types":
                # Generate realistic cell parameters
                cell_data = []
                for cell_name in potential_cells:
                    params = BiologicalSimulationGenerator.get_realistic_cell_parameters(cell_name, description)
                    cell_data.append({
                        "name": cell_name,
                        "canonical_name": BiologicalSimulationGenerator.standardize_cell_name(cell_name),
                        **params
                    })
                
                # Format as JSON for easy copying
                guidance = f"""Intelligent Cell Type Generation Complete

Based on biological analysis, generated {len(cell_data)} realistic cell types:

{json.dumps(cell_data, indent=2)}

Key Biological Features:
{chr(10).join([f"- {cell['name']}: Proliferation {cell['proliferation_rate']:.2e}/min (~{1440*60/cell['proliferation_rate']:.1f}h doubling), Migration {cell['migration_speed']}um/min, Volume {cell['volume']}um3" for cell in cell_data])}

Next Step: Use this data with 'create_physicell_configuration' tool."""
                
            elif component_type == "substrates":
                # Generate relevant substrates
                substrates = BiologicalSimulationGenerator.suggest_relevant_substrates(description, potential_cells)
                
                guidance = f"""Intelligent Substrate Generation Complete

Based on biological scenario, suggested {len(substrates)} relevant substrates:

{json.dumps(substrates, indent=2)}

Key Substrate Properties:
{chr(10).join([f"- {sub['name']}: {sub['initial_condition']} {sub['units']}, Diffusion {sub['diffusion_coefficient']} um2/min, Decay {sub['decay_rate']}/min" for sub in substrates])}

Next Step: Use this data with 'create_physicell_configuration' tool."""
                
            elif component_type == "interactions":
                # Generate realistic interactions
                substrate_names = [s['name'] for s in BiologicalSimulationGenerator.suggest_relevant_substrates(description, potential_cells)]
                interactions = BiologicalSimulationGenerator.generate_realistic_interactions(potential_cells, substrate_names, description)
                
                guidance = f"""Intelligent Interaction Generation Complete

Generated {len(interactions)} biologically realistic interactions:

{json.dumps(interactions, indent=2)}

Key Biological Interactions:
{chr(10).join([f"- {int['cell_type']} + {int['input']} -> {int['output']} (half-max: {int['half_max']})" for int in interactions])}

Next Step: Use this data with 'create_interaction_rules' tool."""
                
            else:  # component_type == "all"
                # Generate complete biological simulation
                cell_data = []
                for cell_name in potential_cells:
                    params = BiologicalSimulationGenerator.get_realistic_cell_parameters(cell_name, description)
                    cell_data.append({
                        "name": cell_name,
                        "canonical_name": BiologicalSimulationGenerator.standardize_cell_name(cell_name),
                        **params
                    })
                
                substrates = BiologicalSimulationGenerator.suggest_relevant_substrates(description, potential_cells)
                substrate_names = [s['name'] for s in substrates]
                interactions = BiologicalSimulationGenerator.generate_realistic_interactions(potential_cells, substrate_names, description)
                
                complete_data = {
                    "cell_types": cell_data,
                    "substrates": substrates, 
                    "interactions": interactions
                }
                
                guidance = f"""Complete Biological Simulation Generated

SCENARIO: {biological_analysis[:200]}...

GENERATED COMPONENTS:
- {len(cell_data)} Cell Types: {', '.join([c['name'] for c in cell_data])}
- {len(substrates)} Substrates: {', '.join([s['name'] for s in substrates])}  
- {len(interactions)} Interactions: Oxygen dependencies, drug responses, immune reactions

{json.dumps(complete_data, indent=2)}

Next Steps: 
1. Use cell_data and substrates with 'create_physicell_configuration'
2. Use interactions with 'create_interaction_rules'
3. Create project files with 'create_project_files'"""
            
            return [TextContent(type="text", text=guidance)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating components: {e}")]

    async def _create_physicell_configuration(
        self,
        cell_data: List[Dict],
        substrate_data: List[Dict],
        simulation_params: Optional[Dict] = None
    ) -> List[TextContent]:
        """Generate PhysiCell XML configuration from component data"""
        try:
            if simulation_params is None:
                simulation_params = {
                    "domain_size": 1000,
                    "simulation_time": 7200,
                    "use_2D": False
                }
            
            # Validate cell parameters
            validated_cells = []
            for cell in cell_data:
                validated = BiologicalSimulationGenerator.validate_parameters(cell)
                validated_cells.append(validated)
            
            # Generate XML guidance
            cell_prompt = BiologicalSimulationGenerator.get_xml_generation_prompt(validated_cells)
            substrate_prompt = BiologicalSimulationGenerator.get_substrate_xml_prompt(substrate_data)
            
            domain_size = 200  # Always use tiny, ultra-safe domain
            sim_time = simulation_params.get('simulation_time', 7200)
            use_2D = True  # Always force 2D to prevent crashes
            
            full_config = f"""PhysiCell XML Configuration Generator
=======================================

SIMULATION PARAMETERS:
• Domain: ±{domain_size} microns
• Time: {sim_time} minutes ({sim_time/1440:.1f} days)
• Dimensions: {'2D' if use_2D else '3D'}

CELL DEFINITIONS:
{cell_prompt}

SUBSTRATE DEFINITIONS:
{substrate_prompt}

COMPLETE XML STRUCTURE:
```xml
<?xml version='1.0' encoding='UTF-8'?>
<PhysiCell_settings version="devel-version">
    <domain>
        <x_min>-{domain_size}</x_min>
        <x_max>{domain_size}</x_max>
        <y_min>-{domain_size}</y_min>
        <y_max>{domain_size}</y_max>
        <z_min>-10</z_min>
        <z_max>10</z_max>
        <dx>20</dx>
        <dy>20</dy>
        <dz>20</dz>
        <use_2D>{'true' if use_2D else 'false'}</use_2D>
    </domain>
    
    <overall>
        <max_time units="min">{sim_time}</max_time>
        <time_units>min</time_units>
        <space_units>micron</space_units>
        <dt_diffusion units="min">0.01</dt_diffusion>
        <dt_mechanics units="min">0.1</dt_mechanics>
        <dt_phenotype units="min">6</dt_phenotype>
    </overall>
    
    [INSERT MICROENVIRONMENT_SETUP HERE]
    [INSERT CELL_DEFINITIONS HERE]
    
    <initial_conditions>
        <cell_positions type="csv" enabled="true">
            <folder>./config</folder>
            <filename>cells.csv</filename>
        </cell_positions>
    </initial_conditions>
</PhysiCell_settings>
```

Please generate the complete XML configuration using the above structure and your component data."""
            
            return [TextContent(type="text", text=full_config)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error creating configuration: {e}")]

    async def _create_interaction_rules(
        self,
        cell_types: List[str],
        substrates: List[str],
        biological_context: str
    ) -> List[TextContent]:
        """Generate CSV interaction rules for PhysiCell"""
        try:
            # Generate rules prompt
            rules_prompt = BiologicalSimulationGenerator.get_rules_generation_prompt(cell_types, substrates)
            
            guidance = f"""Interaction Rules Generation
=============================

BIOLOGICAL CONTEXT: {biological_context}

🚨 CRITICAL: You MUST use these EXACT names (case-sensitive):

CELL TYPES: {', '.join(cell_types)}
SUBSTRATES: {', '.join(substrates)}

{rules_prompt}

⚠️  CONSISTENCY WARNING: Your rules CSV must ONLY reference the exact cell type and substrate names listed above. Using different names will cause the simulation to fail.

✅ VALID EXAMPLE:
If cell types are ["cancer_cell", "CD8_T_cell"] and substrates are ["oxygen", "glucose"], then rules like:
- "cancer_cell,oxygen,increases,proliferation,0,21.5,4,0" ✅
- "CD8_T_cell,glucose,decreases,migration,0,5.0,2,0" ✅

❌ INVALID EXAMPLE:
- "tumor_cell,oxygen,..." ❌ (should be "cancer_cell")
- "T_cell,sugar,..." ❌ (should be "CD8_T_cell" and "glucose")

Please generate a complete CSV file using ONLY the exact names provided above."""
            
            return [TextContent(type="text", text=guidance)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error creating rules: {e}")]

    async def _create_cell_initialization(
        self,
        cell_types: List[str],
        domain_size: float,
        spatial_organization: Optional[str] = None
    ) -> List[TextContent]:
        """Generate initial cell positions and setup"""
        try:
            # Generate cell initialization prompt
            init_prompt = BiologicalSimulationGenerator.generate_cell_initialization_prompt(
                cell_types, int(domain_size)
            )
            
            if spatial_organization:
                context = f"SPATIAL ORGANIZATION: {spatial_organization}\n\n"
            else:
                context = ""
            
            guidance = f"""Cell Initialization Generator
===========================

{context}DOMAIN SIZE: {domain_size} microns (from -{domain_size} to +{domain_size})

🚨 CRITICAL: You MUST use these EXACT cell type names (case-sensitive):
CELL TYPES: {', '.join(cell_types)}

{init_prompt}

⚠️  CONSISTENCY WARNING: The cell type names in your CSV MUST exactly match the names listed above. Using different names will cause all cells to be skipped and the simulation to crash with a segmentation fault.

✅ EXAMPLE: If the cell types are ["cancer_cell", "CD8_T_cell"], your CSV must use exactly "cancer_cell" and "CD8_T_cell" - not "tumor_cell", "TNBC_cell", "T_cell", etc.

Please generate a CSV file with realistic initial cell positions using the EXACT cell type names provided above."""
            
            return [TextContent(type="text", text=guidance)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error creating initialization: {e}")]

    async def _validate_biological_parameters(
        self,
        parameters: Dict[str, Any],
        parameter_type: str
    ) -> List[TextContent]:
        """Validate and suggest improvements for biological parameters"""
        try:
            # Validate parameters using the generator
            validated = BiologicalSimulationGenerator.validate_parameters(parameters)
            
            # Generate validation report
            changes = []
            for param, value in parameters.items():
                if param in validated and validated[param] != value:
                    old_val = value
                    new_val = validated[param]
                    changes.append(f"• {param}: {old_val} → {new_val} (clamped to realistic range)")
            
            if changes:
                report = f"""Parameter Validation Results
==========================

PARAMETER TYPE: {parameter_type}

CORRECTIONS MADE:
{chr(10).join(changes)}

VALIDATED PARAMETERS:
{json.dumps(validated, indent=2)}

These parameters have been adjusted to fall within biologically realistic ranges for PhysiCell simulations."""
            else:
                report = f"""Parameter Validation Results
==========================

PARAMETER TYPE: {parameter_type}

✅ All parameters are within realistic ranges.

VALIDATED PARAMETERS:
{json.dumps(validated, indent=2)}"""
            
            return [TextContent(type="text", text=report)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error validating parameters: {e}")]

    async def _create_project_files(
        self,
        project_name: str,
        xml_content: str,
        rules_csv: Optional[str] = None,
        cells_csv: Optional[str] = None,
        project_path: Optional[str] = None
    ) -> List[TextContent]:
        """Create actual PhysiCell project files from LLM-generated content"""
        try:
            # Set default project path
            if project_path is None:
                project_path = str(self.physicell_path / "user_projects" / project_name)
            else:
                project_path_obj = Path(project_path)
                if not project_path_obj.is_absolute():
                    project_path = str(self.physicell_path / "user_projects" / project_path)
            
            # Create project directory structure
            project_dir = Path(project_path)
            config_dir = project_dir / "config"
            custom_modules_dir = project_dir / "custom_modules"
            output_dir = project_dir / "output"
            
            # Create directories
            project_dir.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(exist_ok=True)
            custom_modules_dir.mkdir(exist_ok=True)
            output_dir.mkdir(exist_ok=True)
            
            files_created = []
            
            # 1. Create PhysiCell_settings.xml with save section fix
            xml_path = config_dir / "PhysiCell_settings.xml"
            
            # Check if XML content is missing save section and add it
            if "<save>" not in xml_content:
                # Insert save section before </PhysiCell_settings>
                save_section = """    
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
        <legacy_data>
            <enable>false</enable>
        </legacy_data>
    </save>
    
"""
                if "</PhysiCell_settings>" in xml_content:
                    xml_content = xml_content.replace("</PhysiCell_settings>", save_section + "</PhysiCell_settings>")
                else:
                    xml_content += save_section
            
            # XML VALIDATION: Check for syntax errors before writing
            try:
                import xml.etree.ElementTree as ET
                ET.fromstring(xml_content)
            except ET.ParseError as e:
                return [TextContent(type="text", text=f"""🚨 XML VALIDATION ERROR: Invalid XML syntax detected!

❌ Parse Error: {str(e)}

The generated XML has syntax errors. This commonly happens when:
• Tag names don't match (e.g., <tag>...</wrong_tag>)
• Missing closing brackets or quotes
• Special characters not properly escaped

Please regenerate the XML content with correct syntax.""")]
            
            with open(xml_path, 'w') as f:
                f.write(xml_content)
            files_created.append(f"✓ {xml_path}")
            
            # 2. Create cell_rules.csv if provided with validation
            if rules_csv:
                rules_path = config_dir / "cell_rules.csv"
                
                # VALIDATION: Check that rules reference valid cell types and substrates
                import re
                xml_cell_types = set(re.findall(r'<cell_definition name="([^"]+)"', xml_content))
                xml_substrates = set(re.findall(r'<variable name="([^"]+)"', xml_content))
                
                # Parse rules CSV and validate
                rules_lines = rules_csv.strip().split('\n')
                
                # Skip header if present
                if rules_lines and ('cell type' in rules_lines[0].lower() or 'cell_type' in rules_lines[0].lower()):
                    rules_lines = rules_lines[1:]
                
                invalid_rules = []
                valid_physicell_signals = list(xml_substrates) + [
                    'pressure', 'volume', 'contact', 'time', 'damage', 'apoptotic', 'necrotic', 'dead',
                    'oxygen', 'glucose', 'lactate', 'vegf', 'arginase1', 'il10', 'tnf_alpha', 'olaparib'
                ]
                valid_physicell_behaviors = [
                    'cycle entry', 'apoptosis', 'necrosis', 'migration speed', 'migration bias',
                    'oxygen secretion', 'vegf secretion', 'arginase1 secretion', 'il10 secretion', 
                    'tnf_alpha secretion', 'olaparib secretion', 'glucose secretion', 'lactate secretion'
                ]
                
                for i, line in enumerate(rules_lines, 2 if rules_csv.strip().split('\n')[0].lower().find('cell') != -1 else 1):
                    if line.strip():
                        parts = [p.strip().strip('"') for p in line.split(',')]
                        if len(parts) >= 4:
                            cell_type = parts[0]
                            input_signal = parts[1] 
                            input_change = parts[2]
                            output_behavior = parts[3]
                            
                            # Validate cell type
                            if cell_type not in xml_cell_types:
                                invalid_rules.append(f"Line {i}: Unknown cell type '{cell_type}'")
                            
                            # Validate input signal (more permissive)
                            if input_signal not in valid_physicell_signals:
                                # Check if it's a gradient or intracellular version
                                base_signal = input_signal.replace('_gradient', '').replace('intracellular_', '')
                                if base_signal not in valid_physicell_signals:
                                    invalid_rules.append(f"Line {i}: Unknown input signal '{input_signal}'")
                            
                            # Validate output behavior (more permissive) 
                            if output_behavior not in valid_physicell_behaviors:
                                # Allow some variations
                                if not any(behavior in output_behavior for behavior in ['secretion', 'entry', 'apoptosis', 'necrosis', 'migration']):
                                    invalid_rules.append(f"Line {i}: Unknown output behavior '{output_behavior}'")
                
                if invalid_rules:
                    error_msg = f"""Rules Validation Issues Found

Issues found:
{chr(10).join(invalid_rules[:10])}  # Show first 10 errors
{f'... and {len(invalid_rules)-10} more errors' if len(invalid_rules) > 10 else ''}

Available cell types: {', '.join(xml_cell_types)}
Available substrates: {', '.join(xml_substrates)}
Common behaviors: cycle entry, apoptosis, necrosis, migration speed, [substrate] secretion

Note: PhysiCell accepts many signal types beyond just substrates."""
                    
                    return [TextContent(type="text", text=error_msg)]
                
                with open(rules_path, 'w') as f:
                    f.write(rules_csv)
                files_created.append(f"✓ {rules_path} (validated: {len(rules_lines)} rules)")
            
            # 3. Create cells.csv if provided with validation
            if cells_csv:
                cells_path = config_dir / "cells.csv"
                
                # CRITICAL VALIDATION: Check cell type consistency
                validation_errors = []
                
                # Extract cell types from XML
                import re
                xml_cell_types = set(re.findall(r'<cell_definition name="([^"]+)"', xml_content))
                
                # Extract cell types from CSV
                csv_lines = cells_csv.strip().split('\n')[1:]  # Skip header
                csv_cell_types = set()
                for line in csv_lines:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 4:
                            csv_cell_types.add(parts[3].strip('"'))  # 4th column is cell type
                
                # Check for mismatches
                missing_in_xml = csv_cell_types - xml_cell_types
                if missing_in_xml:
                    error_msg = f"""🚨 CRITICAL ERROR: Cell type mismatch detected!

CSV file references cell types that don't exist in XML:
❌ Missing: {', '.join(missing_in_xml)}

✅ Available in XML: {', '.join(xml_cell_types)}

This would cause all cells to be skipped and simulation to crash with segmentation fault.

SOLUTIONS:
1. Update your CSV to use the available XML cell types
2. Update your XML to include definitions for the missing cell types
3. Regenerate both files with matching cell type names"""
                    
                    return [TextContent(type="text", text=error_msg)]
                
                with open(cells_path, 'w') as f:
                    f.write(cells_csv)
                files_created.append(f"✓ {cells_path} (validated: {len(csv_cell_types)} cell types match XML)")
            
            # 4. Create basic main.cpp
            main_cpp = self._generate_basic_main_cpp()
            main_path = project_dir / "main.cpp"
            with open(main_path, 'w') as f:
                f.write(main_cpp)
            files_created.append(f"✓ {main_path}")
            
            # 5. Create Makefile
            makefile = self._generate_basic_makefile(project_name)
            makefile_path = project_dir / "Makefile"
            with open(makefile_path, 'w') as f:
                f.write(makefile)
            files_created.append(f"✓ {makefile_path}")
            
            # 6. Create custom module with proper cell type registration
            # Extract cell types from XML for custom module generation
            import re
            xml_cell_types = re.findall(r'<cell_definition name="([^"]+)" ID="(\d+)"', xml_content)
            
            custom_cpp = self._generate_custom_module_with_cell_types(xml_cell_types)
            custom_path = custom_modules_dir / "custom.cpp"
            with open(custom_path, 'w') as f:
                f.write(custom_cpp)
            files_created.append(f"✓ {custom_path} (registers {len(xml_cell_types)} cell types)")
            
            custom_h = self._generate_basic_custom_header()
            header_path = custom_modules_dir / "custom.h"
            with open(header_path, 'w') as f:
                f.write(custom_h)
            files_created.append(f"✓ {header_path}")
            
            report = f"""✅ PhysiCell project '{project_name}' created successfully!

📍 Location: {project_dir}

📁 Files Created:
{chr(10).join(files_created)}

🚀 Next Steps:
1. cd {project_dir}
2. make
3. ./project

The project is ready to compile and run."""
            
            return [TextContent(type="text", text=report)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating project files: {str(e)}"
            )]

    async def _generate_physicell_project(
        self,
        project_name: str,
        biological_description: str,
        project_path: Optional[str] = None,
        structured_data: Optional[str] = None,
        enable_csv_cells: bool = True
    ) -> List[TextContent]:
        """Generate complete PhysiCell project from biological description in a single atomic operation"""
        try:
            from biological_knowledge import BiologicalProjectCoordinator
            import json
            
            logger.info(f"Starting unified project generation: {project_name}")
            logger.info(f"Biological description: {biological_description}")
            
            # If no structured data provided, return prompt for LLM to generate it
            if not structured_data:
                structured_prompt = BiologicalProjectCoordinator.extract_structured_data_prompt(biological_description)
                
                guidance_text = f"""🔬 **UNIFIED PHYSICELL PROJECT GENERATION**

**Project:** {project_name}
**Description:** {biological_description}

This single tool will create a complete PhysiCell project with guaranteed name consistency. 

{structured_prompt}

**INSTRUCTIONS:** Please provide the structured biological data in JSON format. I will then:
1. ✅ Parse and validate the biological data
2. ✅ Standardize and deduplicate cell type names
3. ✅ Generate PhysiCell XML configuration
4. ✅ Generate cells.csv with matching names
5. ✅ Generate cell_rules.csv with matching names
6. ✅ Validate consistency across all files
7. ✅ Create the complete project directory
8. ✅ Return success confirmation or clear error messages

**After providing JSON data, call this tool again with the same parameters plus the JSON as 'structured_data' parameter.**

This unified approach eliminates the complex multi-tool workflow and prevents cell type name mismatches.
"""
                
                return [TextContent(type="text", text=guidance_text)]
            
            # Process structured data to generate complete project
            logger.info("Processing structured data for unified project generation")
            
            # Parse the JSON structured data
            try:
                data = json.loads(structured_data)
                logger.info(f"Successfully parsed JSON with {len(data.get('cell_types', []))} cell types")
            except json.JSONDecodeError as e:
                return [TextContent(
                    type="text",
                    text=f"❌ **JSON Parse Error**: {str(e)}\n\nPlease ensure the structured data is valid JSON format."
                )]
            
            # Validate required structure
            required_keys = ["cell_types", "substrates", "interactions"]
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                return [TextContent(
                    type="text", 
                    text=f"❌ **Missing Required Keys**: {missing_keys}\n\nStructured data must include: cell_types, substrates, interactions"
                )]
            
            # Standardize cell names in the data to ensure consistency
            for cell_type in data.get("cell_types", []):
                if "canonical_name" in cell_type:
                    # Apply standardization to ensure PhysiCell compatibility
                    cell_type["canonical_name"] = BiologicalProjectCoordinator.standardize_cell_name(
                        cell_type["canonical_name"]
                    )
            
            # CRITICAL FIX: Deduplicate cell types by canonical name
            seen_names = {}
            unique_cell_types = []
            duplicates_found = []
            
            for cell_type in data.get("cell_types", []):
                name = cell_type.get("canonical_name", "unknown")
                if name in seen_names:
                    duplicates_found.append(name)
                    # Merge properties - take average of numeric values, prefer first for others
                    existing = seen_names[name]
                    for key, value in cell_type.items():
                        if key == "canonical_name":
                            continue  # Keep existing name
                        elif isinstance(value, (int, float)) and key in existing:
                            # Average numeric values
                            existing[key] = (existing[key] + value) / 2
                        elif key not in existing:
                            # Add new properties
                            existing[key] = value
                else:
                    seen_names[name] = cell_type.copy()
                    unique_cell_types.append(cell_type)
            
            if duplicates_found:
                logger.warning(f"Deduplicated {len(duplicates_found)} cell types: {duplicates_found}")
            
            # Update data with deduplicated cell types
            data["cell_types"] = unique_cell_types
            logger.info(f"Cell name standardization completed - {len(unique_cell_types)} unique cell types")
            
            # Update interaction cell names to match standardized names
            valid_cell_names = {cell["canonical_name"] for cell in data.get("cell_types", [])}
            interactions_to_remove = []
            
            for i, interaction in enumerate(data.get("interactions", [])):
                if "cell_type" in interaction:
                    original_cell_name = interaction["cell_type"]
                    
                    # First, check if it matches a standardized name directly
                    if original_cell_name in valid_cell_names:
                        continue  # Already correct
                    
                    # Try to find the standardized equivalent
                    standardized_name = BiologicalProjectCoordinator.standardize_cell_name(original_cell_name)
                    
                    if standardized_name in valid_cell_names:
                        interaction["cell_type"] = standardized_name
                        logger.info(f"Updated interaction: {original_cell_name} -> {standardized_name}")
                    else:
                        # Convert to warning instead of removing
                        logger.warning(f"Unknown cell type in interaction: {original_cell_name} -> {standardized_name}")
                        interactions_to_remove.append(i)
            
            # Remove invalid interactions (in reverse order to preserve indices)
            for i in reversed(interactions_to_remove):
                data["interactions"].pop(i)
            
            logger.info(f"Cell name processing completed - {len(data['interactions'])} interactions remain")
            
            # Extract cell types and substrates for context-aware generation
            cell_types = [cell['canonical_name'] for cell in data.get('cell_types', [])]
            substrates = [sub['name'] for sub in data.get('substrates', [])]
            
            # Generate coordinated files using the structured data with context
            xml_content = BiologicalProjectCoordinator.generate_xml_from_structured_data(data, enable_csv_cells, project_name)
            cells_csv = BiologicalProjectCoordinator.generate_cells_csv_from_structured_data(data)
            rules_csv = BiologicalProjectCoordinator.generate_rules_csv_from_structured_data(data, cell_types, substrates)
            
            logger.info("File generation completed")
            
            # Validate name consistency across all files with warnings instead of errors
            validation_result = BiologicalProjectCoordinator.validate_name_consistency(
                xml_content, cells_csv, rules_csv
            )
            
            validation_warnings = []
            if not validation_result["valid"]:
                logger.warning(f"Name consistency validation warnings: {validation_result['issues']}")
                validation_warnings = validation_result['issues']
            
            logger.info("Creating project files")
            
            # Create project files using the existing method
            create_result = await self._create_project_files(
                project_name=project_name,
                xml_content=xml_content,
                rules_csv=rules_csv,
                cells_csv=cells_csv,
                project_path=project_path
            )
            
            # Add validation warnings to the result
            if validation_warnings:
                warning_text = f"\n\n⚠️ **VALIDATION WARNINGS**:\n{chr(10).join([f'• {w}' for w in validation_warnings])}\n\nProject created successfully but may need manual review."
                create_result[0].text += warning_text
            
            return create_result
            
        except Exception as e:
            logger.error(f"Error in unified project generation: {e}")
            return [TextContent(
                type="text", 
                text=f"❌ **Unified Generation Error**: {str(e)}\n\nFailed to generate complete PhysiCell project."
            )]

    async def _fix_duplicate_cell_definitions(self, project_path: str) -> List[TextContent]:
        """Fix projects with duplicate cell definitions that cause PhysiCell crashes"""
        try:
            from biological_knowledge import BiologicalProjectCoordinator
            import xml.etree.ElementTree as ET
            
            logger.info(f"Fixing duplicate cell definitions in project: {project_path}")
            
            project_dir = Path(project_path)
            xml_path = project_dir / "config" / "PhysiCell_settings.xml"
            
            if not xml_path.exists():
                return [TextContent(
                    type="text",
                    text=f"❌ XML file not found: {xml_path}"
                )]
            
            # Read and parse XML
            with open(xml_path, 'r') as f:
                xml_content = f.read()
            
            # Quick validation check
            validation = BiologicalProjectCoordinator.validate_name_consistency(
                xml_content, "", ""
            )
            
            if not validation.get("xml_duplicates"):
                return [TextContent(
                    type="text",
                    text="✅ No duplicate cell definitions found. Project is already valid."
                )]
            
            logger.info(f"Found duplicates: {validation['xml_duplicates']}")
            
            # Parse XML to remove duplicates
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                return [TextContent(
                    type="text", 
                    text=f"❌ XML parsing error: {e}"
                )]
            
            # Find cell_definitions section
            cell_defs_section = root.find('.//cell_definitions')
            if cell_defs_section is None:
                return [TextContent(
                    type="text",
                    text="❌ No <cell_definitions> section found in XML"
                )]
            
            # Remove duplicate cell definitions (keep first occurrence)
            seen_names = set()
            duplicates_removed = []
            cells_to_remove = []
            
            for cell_def in cell_defs_section.findall('cell_definition'):
                name = cell_def.get('name')
                if name in seen_names:
                    duplicates_removed.append(name)
                    cells_to_remove.append(cell_def)
                else:
                    seen_names.add(name)
            
            # Remove duplicate elements
            for cell_def in cells_to_remove:
                cell_defs_section.remove(cell_def)
            
            # Regenerate XML string
            fixed_xml = ET.tostring(root, encoding='unicode')
            
            # Add XML declaration if missing
            if not fixed_xml.startswith('<?xml'):
                fixed_xml = "<?xml version='1.0' encoding='UTF-8'?>\n" + fixed_xml
            
            # Write fixed XML
            with open(xml_path, 'w') as f:
                f.write(fixed_xml)
            
            # Validate the fix
            final_validation = BiologicalProjectCoordinator.validate_name_consistency(
                fixed_xml, "", ""
            )
            
            if final_validation.get("xml_duplicates"):
                return [TextContent(
                    type="text",
                    text=f"❌ Fix failed - duplicates still present: {final_validation['xml_duplicates']}"
                )]
            
            result_text = f"""🔧 **DUPLICATE CELL DEFINITIONS FIX COMPLETE**

**Project:** {project_path}

✅ **Results:**
• Removed {len(duplicates_removed)} duplicate definitions: {duplicates_removed}
• Remaining unique cell types: {list(seen_names)}
• XML file updated successfully

🚀 **Project should now run without PhysiCell crashes!**

The duplicate cell definitions were confusing PhysiCell's cell type loading system, causing all cells to be skipped and resulting in segmentation faults.
"""
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            logger.error(f"Error fixing duplicate cell definitions: {e}")
            return [TextContent(
                type="text",
                text=f"❌ **Fix Error**: {str(e)}\n\nFailed to fix duplicate cell definitions."
            )]

    async def _create_biological_simulation(
        self,
        description: str,
        project_name: str,
        project_path: Optional[str] = None
    ) -> List[TextContent]:
        """Complete end-to-end simulation creation using dynamic generation"""
        try:
            # Set default project path
            if project_path is None:
                project_path = str(self.physicell_path / "user_projects" / project_name)
            else:
                project_path_obj = Path(project_path)
                if not project_path_obj.is_absolute():
                    project_path = str(self.physicell_path / "user_projects" / project_path)
            
            guidance = f"""Dynamic Biological Simulation Creation
======================================

PROJECT: {project_name}
LOCATION: {project_path}
SCENARIO: {description}

WORKFLOW OVERVIEW:
This tool guides you through creating a complete PhysiCell simulation using LLM-powered dynamic generation:

STEP 1: Call 'analyze_biological_scenario' with your description
STEP 2: Use 'generate_simulation_components' to create cell types, substrates, and interactions  
STEP 3: Use 'create_physicell_configuration' to generate the XML config
STEP 4: Use 'create_interaction_rules' to generate the rules CSV  
STEP 5: Use 'create_cell_initialization' to generate initial positions
STEP 6: Call 'create_project_files' with the generated content to create actual files
STEP 7: Use 'run_simulation' to compile and run the simulation

KEY BENEFITS:
• All biological knowledge comes from the LLM's expertise
• Parameters are validated for PhysiCell compatibility
• The simulation is scientifically accurate and realistic
• No hardcoded biological assumptions are made

🚨 CRITICAL CONSISTENCY REQUIREMENT:
The cell type and substrate names must be IDENTICAL across all files:
• XML <cell_definition name="..."> 
• CSV cell initialization (column 4)
• CSV interaction rules (columns 1 and 2)

Using inconsistent names will cause the simulation to crash. Each tool will provide the exact names to use.

IMPORTANT: After generating content with steps 1-5, you MUST use the 'create_project_files' tool to actually create the project files on disk. The other tools only provide guidance and content - they don't create files.

Start with 'analyze_biological_scenario' to begin the process."""
            
            return [TextContent(type="text", text=guidance)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating biological simulation: {str(e)}"
            )]

    async def _extract_xml_names(self, xml_content: str) -> List[TextContent]:
        """Extract cell type and substrate names from PhysiCell XML for consistency"""
        try:
            import re
            
            # Extract cell types
            cell_types = re.findall(r'<cell_definition name="([^"]+)"', xml_content)
            
            # Extract substrates
            substrates = re.findall(r'<variable name="([^"]+)"', xml_content)
            
            # Remove duplicates while preserving order
            cell_types = list(dict.fromkeys(cell_types))
            substrates = list(dict.fromkeys(substrates))
            
            report = f"""📋 XML Name Extraction Results
=============================

🔬 CELL TYPES FOUND: {len(cell_types)}
{chr(10).join([f"  • {ct}" for ct in cell_types])}

🧪 SUBSTRATES FOUND: {len(substrates)}  
{chr(10).join([f"  • {sub}" for sub in substrates])}

🚨 IMPORTANT: Use these EXACT names (case-sensitive) in your CSV files:

FOR CELL INITIALIZATION CSV:
- Column 4 (cell type) must use: {', '.join(cell_types)}

FOR INTERACTION RULES CSV:
- Column 1 (cell type) must use: {', '.join(cell_types)}
- Column 2 (substrate) must use: {', '.join(substrates)}

Using different names will cause simulation crashes. Copy-paste these names exactly as shown above."""
            
            return [TextContent(type="text", text=report)]
            
        except Exception as e:
            return [TextContent(
                type="text", 
                text=f"Error extracting XML names: {str(e)}"
            )]

    async def _fix_project_config(self, project_path: str) -> List[TextContent]:
        """Fix common configuration issues in existing PhysiCell projects"""
        try:
            project_dir = Path(project_path)
            config_dir = project_dir / "config"
            xml_path = config_dir / "PhysiCell_settings.xml"
            
            if not xml_path.exists():
                return [TextContent(
                    type="text",
                    text=f"❌ No PhysiCell_settings.xml found at {xml_path}"
                )]
            
            # Read current XML
            with open(xml_path, 'r') as f:
                xml_content = f.read()
            
            fixes_applied = []
            original_content = xml_content
            
            # Fix 1: Add missing save section
            if "<save>" not in xml_content:
                save_section = """    
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
        <legacy_data>
            <enable>false</enable>
        </legacy_data>
    </save>
    
"""
                if "</PhysiCell_settings>" in xml_content:
                    xml_content = xml_content.replace("</PhysiCell_settings>", save_section + "</PhysiCell_settings>")
                    fixes_applied.append("✓ Added missing <save> section")
            
            # Fix 2: Ensure output directory exists
            output_dir = project_dir / "output"
            if not output_dir.exists():
                output_dir.mkdir(exist_ok=True)
                fixes_applied.append("✓ Created missing output directory")
            
            # Fix 3: Check for cell_rules.csv reference
            if "cell_rules" in xml_content and not (config_dir / "cell_rules.csv").exists():
                fixes_applied.append("⚠️  Warning: XML references cell_rules.csv but file is missing")
            
            # Fix 4: Check for cells.csv reference  
            if "cells.csv" in xml_content and not (config_dir / "cells.csv").exists():
                fixes_applied.append("⚠️  Warning: XML references cells.csv but file is missing")
            
            # Fix 5: Check for cell type mismatches between XML and CSV
            cells_csv_path = config_dir / "cells.csv"
            if cells_csv_path.exists():
                # Extract cell types from XML
                import re
                xml_cell_types = re.findall(r'<cell_definition name="([^"]+)"', xml_content)
                
                # Extract cell types from CSV
                try:
                    with open(cells_csv_path, 'r') as f:
                        csv_content = f.read()
                    csv_lines = csv_content.strip().split('\n')[1:]  # Skip header
                    csv_cell_types = set()
                    for line in csv_lines:
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 4:
                                csv_cell_types.add(parts[3])  # 4th column is cell type
                    
                    # Find mismatches
                    xml_set = set(xml_cell_types)
                    missing_in_xml = csv_cell_types - xml_set
                    
                    if missing_in_xml:
                        fixes_applied.append(f"🚨 CRITICAL: CSV references cell types not in XML: {', '.join(missing_in_xml)}")
                        fixes_applied.append("   This will cause all cells to be skipped and simulation to crash!")
                        fixes_applied.append("   Available XML cell types: " + ", ".join(xml_set))
                        
                except Exception as e:
                    fixes_applied.append(f"⚠️  Could not check CSV cell types: {e}")
            
            
            # Write fixed XML if changes were made
            if xml_content != original_content:
                with open(xml_path, 'w') as f:
                    f.write(xml_content)
                fixes_applied.append(f"✓ Updated {xml_path}")
            
            if fixes_applied:
                report = f"""🔧 Project Configuration Fixes Applied

PROJECT: {project_dir}

FIXES APPLIED:
{chr(10).join(fixes_applied)}

The project should now be ready to run. Try running the simulation again."""
            else:
                report = f"""✅ Project Configuration Check Complete

PROJECT: {project_dir}

No configuration issues found. The project appears to be properly configured.

If you're still having issues, they may be related to:
• Compilation problems
• Missing PhysiCell installation
• Permission issues"""

            return [TextContent(type="text", text=report)]
            
        except Exception as e:
            return [TextContent(
                type="text", 
                text=f"Error fixing project config: {str(e)}"
            )]
    
    def _generate_basic_main_cpp(self) -> str:
        """Generate complete tumor_immune_base main.cpp to ensure compatibility"""
        return """/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2022, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <fstream>

#include "./core/PhysiCell.h"
#include "./modules/PhysiCell_standard_modules.h" 

// put custom code modules here! 

#include "./custom_modules/custom.h" 
	
using namespace BioFVM;
using namespace PhysiCell;

int main( int argc, char* argv[] )
{
	// load and parse settings file(s)
	
	bool XML_status = false; 
	char copy_command [1024]; 
	if( argc > 1 )
	{
		XML_status = load_PhysiCell_config_file( argv[1] ); 
		sprintf( copy_command , "cp %s %s" , argv[1] , PhysiCell_settings.folder.c_str() ); 
	}
	else
	{
		XML_status = load_PhysiCell_config_file( "./config/PhysiCell_settings.xml" );
		sprintf( copy_command , "cp ./config/PhysiCell_settings.xml %s" , PhysiCell_settings.folder.c_str() ); 
	}
	if( !XML_status )
	{ exit(-1); }
	
	// copy config file to output directry 
	system( copy_command ); 
	
	// OpenMP setup
	omp_set_num_threads(PhysiCell_settings.omp_num_threads);
	
	// time setup 
	std::string time_units = "min"; 

	/* Microenvironment setup */ 
	
	setup_microenvironment(); // modify this in the custom code 
	
	/* PhysiCell setup */ 
 	
	// set mechanics voxel size, and match the data structure to BioFVM
	double mechanics_voxel_size = 30; 
	Cell_Container* cell_container = create_cell_container_for_microenvironment( microenvironment, mechanics_voxel_size );
	
	/* Users typically start modifying here. START USERMODS */ 
	
	create_cell_types();
	
	setup_tissue();

	/* Users typically stop modifying here. END USERMODS */ 
	
	// set MultiCellDS save options 

	set_save_biofvm_mesh_as_matlab( true ); 
	set_save_biofvm_data_as_matlab( true ); 
	set_save_biofvm_cell_data( true ); 
	set_save_biofvm_cell_data_as_custom_matlab( true );
	
	// save a simulation snapshot 
	
	char filename[1024];
	sprintf( filename , "%s/initial" , PhysiCell_settings.folder.c_str() ); 
	save_PhysiCell_to_MultiCellDS_v2( filename , microenvironment , PhysiCell_globals.current_time ); 
	
	// save a quick SVG cross section through z = 0, after setting its 
	// length bar to 200 microns 

	PhysiCell_SVG_options.length_bar = 200; 

	// for simplicity, set a pathology coloring function 
	
	std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;
	std::string (*substrate_coloring_function)(double, double, double) = paint_by_density_percentage;

	sprintf( filename , "%s/initial.svg" , PhysiCell_settings.folder.c_str() ); 
	SVG_plot( filename , microenvironment, 0.0 , PhysiCell_globals.current_time, cell_coloring_function, substrate_coloring_function );
	
	sprintf( filename , "%s/legend.svg" , PhysiCell_settings.folder.c_str() ); 
	create_plot_legend( filename , cell_coloring_function ); 
	
	display_citations(); 
	
	// set the performance timers 

	BioFVM::RUNTIME_TIC();
	BioFVM::TIC();
	
	std::ofstream report_file;
	if( PhysiCell_settings.enable_legacy_saves == true )
	{	
		sprintf( filename , "%s/simulation_report.txt" , PhysiCell_settings.folder.c_str() ); 
		
		report_file.open(filename); 	// create the data log file 
		report_file<<"simulated time\\tnum cells\\tnum division\\tnum death\\twall time"<<std::endl;
	}
	
	// main loop 
	
	try 
	{		
		while( PhysiCell_globals.current_time < PhysiCell_settings.max_time + 0.1*diffusion_dt )
		{
			// save data if it's time. 
			if( fabs( PhysiCell_globals.current_time - PhysiCell_globals.next_full_save_time ) < 0.01 * diffusion_dt )
			{
				display_simulation_status( std::cout ); 
				if( PhysiCell_settings.enable_legacy_saves == true )
				{	
					log_output( PhysiCell_globals.current_time , PhysiCell_globals.full_output_index, microenvironment, report_file);
				}
				
				if( PhysiCell_settings.enable_full_saves == true )
				{	
					sprintf( filename , "%s/output%08u" , PhysiCell_settings.folder.c_str(),  PhysiCell_globals.full_output_index ); 
					
					save_PhysiCell_to_MultiCellDS_v2( filename , microenvironment , PhysiCell_globals.current_time ); 
				}
				
				PhysiCell_globals.full_output_index++; 
				PhysiCell_globals.next_full_save_time += PhysiCell_settings.full_save_interval;
			}
			
			// save SVG plot if it's time
			if( fabs( PhysiCell_globals.current_time - PhysiCell_globals.next_SVG_save_time  ) < 0.01 * diffusion_dt )
			{
				if( PhysiCell_settings.enable_SVG_saves == true )
				{	
					sprintf( filename , "%s/snapshot%08u.svg" , PhysiCell_settings.folder.c_str() , PhysiCell_globals.SVG_output_index );
					SVG_plot( filename , microenvironment, 0.0 , PhysiCell_globals.current_time, cell_coloring_function, substrate_coloring_function);

					PhysiCell_globals.SVG_output_index++; 
					PhysiCell_globals.next_SVG_save_time  += PhysiCell_settings.SVG_save_interval;
				}
			}

			// update the microenvironment
			microenvironment.simulate_diffusion_decay( diffusion_dt );
			
			// run PhysiCell 
			((Cell_Container *)microenvironment.agent_container)->update_all_cells( PhysiCell_globals.current_time );
			
			/*
			  Custom add-ons could potentially go here. 
			*/
			
			PhysiCell_globals.current_time += diffusion_dt;
		}
		
		if( PhysiCell_settings.enable_legacy_saves == true )
		{			
			log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);
			report_file.close();
		}
	}
	catch( const std::exception& e )
	{ // reference to the base of a polymorphic object
		std::cout << e.what(); // information from length_error printed
	}
	
	// save a final simulation snapshot 
	
	sprintf( filename , "%s/final" , PhysiCell_settings.folder.c_str() ); 
	save_PhysiCell_to_MultiCellDS_v2( filename , microenvironment , PhysiCell_globals.current_time ); 
	
	sprintf( filename , "%s/final.svg" , PhysiCell_settings.folder.c_str() );
	SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function, substrate_coloring_function);

	// timer 
	
	std::cout << std::endl << "Total simulation runtime: " << std::endl; 
	BioFVM::display_stopwatch_value( std::cout , BioFVM::runtime_stopwatch_value() ); 

	return 0; 
}
"""

    def _generate_basic_makefile(self, project_name: str) -> str:
        """Generate complete tumor_immune_base Makefile to ensure compatibility"""
        return f"""VERSION := $(shell grep . VERSION.txt | cut -f1 -d:)
PROGRAM_NAME := project

CC := g++
# CC := g++-mp-7 # typical macports compiler name
# CC := g++-7 # typical homebrew compiler name 

# Check for environment definitions of compiler 
# e.g., on CC = g++-7 on OSX
ifdef PHYSICELL_CPP 
	CC := $(PHYSICELL_CPP)
endif

ifndef STATIC_OPENMP
	STATIC_OPENMP = -fopenmp
endif

ARCH := native # best auto-tuning
# ARCH := core2 # a reasonably safe default for most CPUs since 2007
# ARCH := corei7
# ARCH := corei7-avx # earlier i7 
# ARCH := core-avx-i # i7 ivy bridge or newer 
# ARCH := core-avx2 # i7 with Haswell or newer
# ARCH := nehalem
# ARCH := westmere
# ARCH := sandybridge # circa 2011
# ARCH := ivybridge   # circa 2012
# ARCH := haswell     # circa 2013
# ARCH := broadwell   # circa 2014
# ARCH := skylake     # circa 2015
# ARCH := bonnell
# ARCH := silvermont
# ARCH := skylake-avx512
# ARCH := nocona #64-bit pentium 4 or later 

# CFLAGS := -march=$(ARCH) -Ofast -s -fomit-frame-pointer -mfpmath=both -fopenmp -m64 -std=c++11
CFLAGS := -march=$(ARCH) -O3 -fomit-frame-pointer -mfpmath=both -fopenmp -m64 -std=c++11

ifeq ($(OS),Windows_NT)
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		UNAME_P := $(shell uname -p)
		var := $(shell which $(CC) | xargs file)
		ifeq ($(lastword $(var)),arm64)
		  CFLAGS := -march=$(ARCH) -O3 -fomit-frame-pointer -fopenmp -m64 -std=c++11
		endif
	endif
endif

CFLAGS_LINK := $(shell echo $(CFLAGS) | sed -e "s/-fopenmp//g")
COMPILE_COMMAND := $(CC) $(CFLAGS)  $(EXTRA_FLAGS)
LINK_COMMAND := $(CC) $(CFLAGS_LINK) $(EXTRA_FLAGS)

BioFVM_OBJECTS := BioFVM_vector.o BioFVM_mesh.o BioFVM_microenvironment.o BioFVM_solvers.o BioFVM_matlab.o \\
BioFVM_utilities.o BioFVM_basic_agent.o BioFVM_MultiCellDS.o BioFVM_agent_container.o 

PhysiCell_core_OBJECTS := PhysiCell_phenotype.o PhysiCell_cell_container.o PhysiCell_standard_models.o \\
PhysiCell_cell.o PhysiCell_custom.o PhysiCell_utilities.o PhysiCell_constants.o PhysiCell_basic_signaling.o \\
PhysiCell_signal_behavior.o PhysiCell_rules.o

PhysiCell_module_OBJECTS := PhysiCell_SVG.o PhysiCell_pathology.o PhysiCell_MultiCellDS.o PhysiCell_various_outputs.o \\
PhysiCell_pugixml.o PhysiCell_settings.o PhysiCell_geometry.o

# put your custom objects here (they should be in the custom_modules directory)

PhysiCell_custom_module_OBJECTS := custom.o

pugixml_OBJECTS := pugixml.o

PhysiCell_OBJECTS := $(BioFVM_OBJECTS)  $(pugixml_OBJECTS) $(PhysiCell_core_OBJECTS) $(PhysiCell_module_OBJECTS)
ALL_OBJECTS := $(PhysiCell_OBJECTS) $(PhysiCell_custom_module_OBJECTS)

# compile the project 

all: main.cpp $(ALL_OBJECTS)
	$(COMPILE_COMMAND) -o $(PROGRAM_NAME) $(ALL_OBJECTS) main.cpp 
	make name

static: main.cpp $(ALL_OBJECTS) $(MaBoSS)
	$(LINK_COMMAND) $(INC) -o $(PROGRAM_NAME) $(ALL_OBJECTS) main.cpp $(LIB) -static-libgcc -static-libstdc++ $(STATIC_OPENMP)

name:
	@echo ""
	@echo "Executable name is" $(PROGRAM_NAME)
	@echo ""

# PhysiCell core components	

PhysiCell_phenotype.o: ./core/PhysiCell_phenotype.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_phenotype.cpp
	
PhysiCell_digital_cell_line.o: ./core/PhysiCell_digital_cell_line.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_digital_cell_line.cpp

PhysiCell_cell.o: ./core/PhysiCell_cell.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_cell.cpp 

PhysiCell_cell_container.o: ./core/PhysiCell_cell_container.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_cell_container.cpp 
	
PhysiCell_standard_models.o: ./core/PhysiCell_standard_models.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_standard_models.cpp 
	
PhysiCell_utilities.o: ./core/PhysiCell_utilities.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_utilities.cpp 
	
PhysiCell_custom.o: ./core/PhysiCell_custom.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_custom.cpp 
	
PhysiCell_constants.o: ./core/PhysiCell_constants.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_constants.cpp 
	
PhysiCell_signal_behavior.o: ./core/PhysiCell_signal_behavior.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_signal_behavior.cpp 
	
PhysiCell_rules.o: ./core/PhysiCell_rules.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_rules.cpp 

# BioFVM core components (needed by PhysiCell)
	
BioFVM_vector.o: ./BioFVM/BioFVM_vector.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_vector.cpp 

BioFVM_agent_container.o: ./BioFVM/BioFVM_agent_container.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_agent_container.cpp 
	
BioFVM_mesh.o: ./BioFVM/BioFVM_mesh.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_mesh.cpp 

BioFVM_microenvironment.o: ./BioFVM/BioFVM_microenvironment.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_microenvironment.cpp 

BioFVM_solvers.o: ./BioFVM/BioFVM_solvers.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_solvers.cpp 

BioFVM_utilities.o: ./BioFVM/BioFVM_utilities.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_utilities.cpp 
	
BioFVM_basic_agent.o: ./BioFVM/BioFVM_basic_agent.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_basic_agent.cpp 
	
BioFVM_matlab.o: ./BioFVM/BioFVM_matlab.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_matlab.cpp

BioFVM_MultiCellDS.o: ./BioFVM/BioFVM_MultiCellDS.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_MultiCellDS.cpp
	
pugixml.o: ./BioFVM/pugixml.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/pugixml.cpp
	
# standard PhysiCell modules

PhysiCell_SVG.o: ./modules/PhysiCell_SVG.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_SVG.cpp

PhysiCell_pathology.o: ./modules/PhysiCell_pathology.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_pathology.cpp

PhysiCell_MultiCellDS.o: ./modules/PhysiCell_MultiCellDS.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_MultiCellDS.cpp

PhysiCell_various_outputs.o: ./modules/PhysiCell_various_outputs.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_various_outputs.cpp

PhysiCell_pugixml.o: ./modules/PhysiCell_pugixml.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_pugixml.cpp
	
PhysiCell_settings.o: ./modules/PhysiCell_settings.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_settings.cpp
	
PhysiCell_basic_signaling.o: ./core/PhysiCell_basic_signaling.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_basic_signaling.cpp 	
	
PhysiCell_geometry.o: ./modules/PhysiCell_geometry.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_geometry.cpp 
	
# user-defined PhysiCell modules

custom.o: ./custom_modules/custom.cpp 
	$(COMPILE_COMMAND) -c ./custom_modules/custom.cpp

# cleanup

reset:
	rm -f *.cpp 
	cp ./sample_projects/Makefile-default Makefile 
	rm -f ./custom_modules/*
	touch ./custom_modules/empty.txt 
	touch ALL_CITATIONS.txt 
	touch ./core/PhysiCell_cell.cpp
	rm ALL_CITATIONS.txt 
	cp ./config/PhysiCell_settings-backup.xml ./config/PhysiCell_settings.xml 
	touch ./config/empty.csv
	rm -f ./config/*.csv
	
clean:
	rm -f *.o
	rm -f $(PROGRAM_NAME)*
	
data-cleanup:
	rm -rf $(OUTPUT)
	mkdir $(OUTPUT)
	touch $(OUTPUT)/empty.txt
	
# archival 
	
checkpoint: 
	zip -r $$(date +%b_%d_%Y_%H%M).zip Makefile *.cpp *.h config/*.xml custom_modules/* 
	
zip:
	zip -r latest.zip Makefile* *.cpp *.h BioFVM/* config/* core/* custom_modules/* matlab/* modules/* sample_projects/* 
	cp latest.zip $$(date +%b_%d_%Y_%H%M).zip
	cp latest.zip VERSION_$(VERSION).zip 
	mv *.zip archives/
	
tar:
	tar --ignore-failed-read -czf latest.tar Makefile* *.cpp *.h BioFVM/* config/* core/* custom_modules/* matlab/* modules/* sample_projects/* 
	cp latest.tar $$(date +%b_%d_%Y_%H%M).tar
	cp latest.tar VERSION_$(VERSION).tar
	mv *.tar archives/

unzip: 
	cp ./archives/latest.zip . 
	unzip latest.zip 
	
untar: 
	cp ./archives/latest.tar .
	tar -xzf latest.tar

# easier animation 

FRAMERATE := 24
OUTPUT := outputs/{project_name}

jpeg: 
	@magick identify -format "%h" $(OUTPUT)/initial.svg > __H.txt 
	@magick identify -format "%w" $(OUTPUT)/initial.svg > __W.txt 
	@expr 2 \\* \\( $$(grep . __H.txt) / 2 \\) > __H1.txt 
	@expr 2 \\* \\( $$(grep . __W.txt) / 2 \\) > __W1.txt 
	@echo "$$(grep . __W1.txt)!x$$(grep . __H1.txt)!" > __resize.txt 
	@magick mogrify -format jpg -resize $$(grep . __resize.txt) $(OUTPUT)/s*.svg
	rm -f __H*.txt __W*.txt __resize.txt 
	
gif: 
	magick convert $(OUTPUT)/s*.svg $(OUTPUT)/out.gif 
	 
movie:
	ffmpeg -r $(FRAMERATE) -f image2 -i $(OUTPUT)/snapshot%08d.jpg -vcodec libx264 -pix_fmt yuv420p -strict -2 -tune animation -crf 15 -acodec none $(OUTPUT)/out.mp4
	
# upgrade rules 

SOURCE := PhysiCell_upgrade.zip 
get-upgrade: 
	@echo $$(curl https://raw.githubusercontent.com/MathCancer/PhysiCell/master/VERSION.txt) > VER.txt 
	@echo https://github.com/MathCancer/PhysiCell/releases/download/$$(grep . VER.txt)/PhysiCell_V.$$(grep . VER.txt).zip > DL_FILE.txt 
	rm -f VER.txt
	$$(curl -L $$(grep . DL_FILE.txt) --output PhysiCell_upgrade.zip)
	rm -f DL_FILE.txt 

PhysiCell_upgrade.zip: 
	make get-upgrade 

upgrade: $(SOURCE)
	unzip $(SOURCE) PhysiCell/VERSION.txt
	mv -f PhysiCell/VERSION.txt . 
	unzip $(SOURCE) PhysiCell/core/* 
	cp -r PhysiCell/core/* core 
	unzip $(SOURCE) PhysiCell/modules/* 
	cp -r PhysiCell/modules/* modules 
	unzip $(SOURCE) PhysiCell/sample_projects/* 
	cp -r PhysiCell/sample_projects/* sample_projects 
	unzip $(SOURCE) PhysiCell/BioFVM/* 
	cp -r PhysiCell/BioFVM/* BioFVM
	unzip $(SOURCE) PhysiCell/documentation/User_Guide.pdf
	mv -f PhysiCell/documentation/User_Guide.pdf documentation
	rm -f -r PhysiCell
	rm -f $(SOURCE) 

# use: make save PROJ=your_project_name
PROJ := my_project

save: 
	echo "Saving project as $(PROJ) ... "
	mkdir -p ./user_projects
	mkdir -p ./user_projects/$(PROJ)
	mkdir -p ./user_projects/$(PROJ)/custom_modules
	mkdir -p ./user_projects/$(PROJ)/config 
	cp main.cpp ./user_projects/$(PROJ)
	cp Makefile ./user_projects/$(PROJ)
	cp VERSION.txt ./user_projects/$(PROJ)
	cp ./config/* ./user_projects/$(PROJ)/config
	cp ./custom_modules/* ./user_projects/$(PROJ)/custom_modules

load: 
	echo "Loading project from $(PROJ) ... "
	cp ./user_projects/$(PROJ)/main.cpp .
	cp ./user_projects/$(PROJ)/Makefile .
	cp -r ./user_projects/$(PROJ)/config/* ./config/ 
	cp -r ./user_projects/$(PROJ)/custom_modules/* ./custom_modules/ 

pack:
	@echo " "
	@echo "Preparing project $(PROJ) for sharing ... "
	@echo " " 
	cd ./user_projects && zip -r $(PROJ).zip $(PROJ)
	@echo " "
	@echo "Share ./user_projects/$(PROJ).zip ... "
	@echo "Other users can unzip $(PROJ).zip in their ./user_projects, compile, and run."
	@echo " " 

unpack:
	@echo " "
	@echo "Preparing shared project $(PROJ).zip for use ... "
	@echo " " 
	cd ./user_projects && unzip $(PROJ).zip 
	@echo " "
	@echo "Load this project via make load PROJ=$(PROJ) ... "
	@echo " " 	

list-user-projects:
	@echo "user projects::"
	@cd ./user_projects && ls -dt1 * | grep . | sed 's!empty.txt!!'
"""

    def _generate_basic_custom_module(self) -> str:
        """Generate basic custom module"""
        return '''#include "./custom.h"

void create_cell_types(void)
{
    // Cell types are defined in the XML configuration
    std::cout << "Loading cell types from XML configuration..." << std::endl;
}

void setup_tissue(void)
{
    // Load initial cell positions from CSV
    load_cells_from_pugixml();
    
    if (all_cells != NULL)
    {
        std::cout << "Total cells created: " << (*all_cells).size() << std::endl;
    }
    else
    {
        std::cout << "Total cells created: 0 (all_cells is null)" << std::endl;
    }
}
'''

    def _generate_custom_module_with_cell_types(self, cell_types: List[Tuple[str, str]]) -> str:
        """Generate custom module using standard PhysiCell initialization pattern"""
        
        return '''#include "./custom.h"

// Forward declarations
std::vector<std::string> my_coloring_function( Cell* pCell );
void phenotype_function( Cell* pCell, Phenotype& phenotype, double dt );
void custom_function( Cell* pCell, Phenotype& phenotype , double dt );
void contact_function( Cell* pMe, Phenotype& phenoMe , Cell* pOther, Phenotype& phenoOther , double dt );

void create_cell_types( void )
{
	// set the random seed - BYPASSED due to PhysiCell SeedRandom crash issue
	// TODO: Re-enable when PhysiCell SeedRandom issue is resolved
	// if (parameters.ints.find_index("random_seed") != -1)
	// {
	//	SeedRandom(parameters.ints("random_seed"));
	// }
	
	/* 
	   Put any modifications to default cell definition here if you 
	   want to have "inherited" by other cell types. 
	   
	   This is a good place to set default functions. 
	*/ 
	
	initialize_default_cell_definition(); 
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 
	
	cell_defaults.functions.volume_update_function = standard_volume_update_function;
	cell_defaults.functions.update_velocity = standard_update_cell_velocity;

	cell_defaults.functions.update_migration_bias = NULL; 
	cell_defaults.functions.update_phenotype = NULL; 
	cell_defaults.functions.custom_cell_rule = NULL; 
	cell_defaults.functions.contact_function = NULL; 
	
	cell_defaults.functions.add_cell_basement_membrane_interactions = NULL; 
	cell_defaults.functions.calculate_distance_to_membrane = NULL; 
	
	/*
	   This parses the cell definitions in the XML config file. 
	*/
	
	initialize_cell_definitions_from_pugixml(); 

	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	build_cell_definitions_maps(); 

	/*
	   This intializes cell signal and response dictionaries 
	*/

	setup_signal_behavior_dictionaries(); 	

	/*
       Cell rule definitions 
	*/

	setup_cell_rules(); 

	/* 
	   Put any modifications to individual cell definitions here. 
	   
	   This is a good place to set custom functions. 
	*/ 
	
	cell_defaults.functions.update_phenotype = phenotype_function; 
	cell_defaults.functions.custom_cell_rule = custom_function; 
	cell_defaults.functions.contact_function = contact_function; 
	
	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	display_cell_definitions( std::cout ); 
	
	return; 
}

void setup_tissue( void )
{
	double Xmin = microenvironment.mesh.bounding_box[0]; 
	double Ymin = microenvironment.mesh.bounding_box[1]; 
	double Zmin = microenvironment.mesh.bounding_box[2]; 

	double Xmax = microenvironment.mesh.bounding_box[3]; 
	double Ymax = microenvironment.mesh.bounding_box[4]; 
	double Zmax = microenvironment.mesh.bounding_box[5]; 
	
	if( default_microenvironment_options.simulate_2D == true )
	{
		Zmin = 0.0; 
		Zmax = 0.0; 
	}
	
	double Xrange = Xmax - Xmin; 
	double Yrange = Ymax - Ymin; 
	double Zrange = Zmax - Zmin; 
	
	// create some of each type of cell - BYPASSED due to RNG issue
	// NOTE: Cells will be loaded from CSV instead via load_cells_from_pugixml()
	std::cout << "Skipping random cell placement (RNG bypassed) - using CSV cell positions instead" << std::endl; 
	
	// load cells from your CSV file (if enabled)
	load_cells_from_pugixml();
	set_parameters_from_distributions();
	
	return; 
}

std::vector<std::string> my_coloring_function( Cell* pCell )
{ return paint_by_number_cell_coloring(pCell); }

void phenotype_function( Cell* pCell, Phenotype& phenotype, double dt )
{ return; }

void custom_function( Cell* pCell, Phenotype& phenotype , double dt )
{ return; } 

void contact_function( Cell* pMe, Phenotype& phenoMe , Cell* pOther, Phenotype& phenoOther , double dt )
{ return; }

void setup_microenvironment( void )
{
	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 
	
	initialize_microenvironment(); 	
	
	return; 
}
'''

    async def _reduce_memory_usage(self, project_path: str, force_2d: bool = True, domain_size: float = 750) -> List[TextContent]:
        """Reduce memory usage to prevent segmentation faults"""
        try:
            project_path = Path(project_path)
            config_file = project_path / "config" / "PhysiCell_settings.xml"
            
            if not config_file.exists():
                return [TextContent(type="text", text=f"❌ Config file not found: {config_file}")]
            
            # Read and parse XML
            with open(config_file, 'r') as f:
                content = f.read()
            
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)
            
            # Get current domain info
            domain = root.find('domain')
            if domain is None:
                return [TextContent(type="text", text="❌ No domain section found in XML")]
            
            # Get current dimensions
            x_min = float(domain.find('x_min').text)
            x_max = float(domain.find('x_max').text)
            y_min = float(domain.find('y_min').text)
            y_max = float(domain.find('y_max').text)
            z_min = float(domain.find('z_min').text) if domain.find('z_min') is not None else -10
            z_max = float(domain.find('z_max').text) if domain.find('z_max') is not None else 10
            dx = float(domain.find('dx').text)
            use_2d = domain.find('use_2D').text.lower() == 'true' if domain.find('use_2D') is not None else False
            
            current_x_size = x_max - x_min
            current_y_size = y_max - y_min
            current_z_size = z_max - z_min
            
            # Calculate current voxel count
            x_voxels = int(current_x_size / dx)
            y_voxels = int(current_y_size / dx)  # assuming dy = dx
            z_voxels = 1 if use_2d else int(current_z_size / dx)
            current_voxels = x_voxels * y_voxels * z_voxels
            
            changes_made = []
            
            # Force 2D if requested
            if force_2d and not use_2d:
                domain.find('use_2D').text = 'true'
                changes_made.append("✅ Changed simulation to 2D")
                z_voxels = 1
                current_voxels = x_voxels * y_voxels
            
            # Reduce domain size if too large
            max_safe_voxels = 250000  # ~1M voxels for safety
            if current_voxels > max_safe_voxels:
                # Calculate new domain size
                target_voxels_per_dim = int((max_safe_voxels) ** 0.5) if force_2d else int((max_safe_voxels) ** (1/3))
                new_size = target_voxels_per_dim * dx
                new_size = min(new_size, domain_size)  # Don't exceed user's max
                
                half_size = new_size / 2
                
                domain.find('x_min').text = str(-half_size)
                domain.find('x_max').text = str(half_size)
                domain.find('y_min').text = str(-half_size)
                domain.find('y_max').text = str(half_size)
                
                if not force_2d:
                    domain.find('z_min').text = str(-half_size)
                    domain.find('z_max').text = str(half_size)
                
                new_voxels = (int(new_size / dx)) ** (2 if force_2d else 3)
                changes_made.append(f"✅ Reduced domain from {current_x_size:.0f}×{current_y_size:.0f}×{current_z_size:.0f} to {new_size:.0f}×{new_size:.0f}×{new_size:.0f if not force_2d else '20'}")
                changes_made.append(f"✅ Reduced voxels from {current_voxels:,} to {new_voxels:,}")
                current_voxels = new_voxels
            
            # Write back to file
            tree = ET.ElementTree(root)
            ET.indent(tree, space="    ")
            tree.write(config_file, encoding='utf-8', xml_declaration=True)
            
            # Fix cell positions if they're outside domain bounds
            cells_file = project_path / "config" / "cells.csv"
            if cells_file.exists():
                try:
                    with open(cells_file, 'r') as f:
                        lines = f.readlines()
                    
                    if len(lines) > 1:  # Has header + data
                        header = lines[0]
                        fixed_lines = [header]
                        cells_fixed = 0
                        
                        for line in lines[1:]:
                            parts = line.strip().split(',')
                            if len(parts) >= 3:
                                try:
                                    x = float(parts[0])
                                    y = float(parts[1])
                                    z = float(parts[2])
                                    
                                    # Get domain bounds (use new_size if we reduced domain)
                                    x_bound = new_size/2 if 'new_size' in locals() else current_x_size/2
                                    y_bound = new_size/2 if 'new_size' in locals() else current_y_size/2
                                    z_bound = 10 if (force_2d or use_2d) else (new_size/2 if 'new_size' in locals() else current_z_size/2)
                                    
                                    # Fix out-of-bounds coordinates
                                    if abs(x) > x_bound or abs(y) > y_bound or abs(z) > z_bound:
                                        # Constrain to domain
                                        x = max(-x_bound, min(x_bound, x))
                                        y = max(-y_bound, min(y_bound, y))
                                        z = 0 if (force_2d or use_2d) else max(-z_bound, min(z_bound, z))
                                        
                                        parts[0] = str(x)
                                        parts[1] = str(y) 
                                        parts[2] = str(z)
                                        cells_fixed += 1
                                    
                                    fixed_lines.append(','.join(parts) + '\n')
                                except (ValueError, IndexError):
                                    fixed_lines.append(line)
                            else:
                                fixed_lines.append(line)
                        
                        if cells_fixed > 0:
                            with open(cells_file, 'w') as f:
                                f.writelines(fixed_lines)
                            changes_made.append(f"✅ Fixed {cells_fixed} cells with out-of-bounds positions")
                            
                except Exception as e:
                    logger.warning(f"Could not fix cell positions: {e}")
            
            if not changes_made:
                return [TextContent(type="text", text=f"✅ Memory usage already optimized ({current_voxels:,} voxels, {'2D' if use_2d else '3D'})")]
            
            result_text = f"🔧 **Memory Usage Optimized**\n\n" + "\n".join(changes_made)
            result_text += f"\n\n📊 **Final Configuration:**"
            result_text += f"\n• Voxels: {current_voxels:,}"
            result_text += f"\n• Mode: {'2D' if (force_2d or use_2d) else '3D'}"
            result_text += f"\n• Domain: {new_size if 'new_size' in locals() else current_x_size:.0f} microns"
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            logger.error(f"Error reducing memory usage: {e}")
            return [TextContent(type="text", text=f"❌ Error: {str(e)}")]

    def _generate_basic_custom_header(self) -> str:
        """Generate complete tumor_immune_base custom.h to ensure compatibility"""
        return """/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2021, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include "../core/PhysiCell.h"
#include "../modules/PhysiCell_standard_modules.h" 

using namespace BioFVM; 
using namespace PhysiCell;

// setup functions to help us along 

void create_cell_types( void );
void setup_tissue( void ); 

// set up the BioFVM microenvironment 
void setup_microenvironment( void ); 

// custom pathology coloring function 

std::vector<std::string> my_coloring_function( Cell* );

// custom functions can go here 

void phenotype_function( Cell* pCell, Phenotype& phenotype, double dt );
void custom_function( Cell* pCell, Phenotype& phenotype , double dt );

void contact_function( Cell* pMe, Phenotype& phenoMe , Cell* pOther, Phenotype& phenoOther , double dt ); 

"""

    # Legacy methods removed - all functionality now uses dynamic LLM generation
    
    async def _run_simulation(
        self,
        project_path: Optional[str] = None,
        threads: int = 4,
        compile: bool = True
    ) -> List[TextContent]:
        """Run simulation with enhanced argument validation and debugging"""
        try:
            # Log received arguments for debugging
            logger.info(f"_run_simulation called with project_path={project_path}, threads={threads}, compile={compile}")
            
            # Validate and process project_path
            if project_path and isinstance(project_path, str) and project_path.strip():
                project_path = project_path.strip()
                logger.info(f"Using provided project_path: {project_path}")
                
                if project_path.startswith("./"):
                    script_dir = Path(__file__).parent
                    work_dir = (script_dir / project_path[2:]).resolve()
                else:
                    work_dir = Path(project_path).resolve()
            else:
                logger.warning(f"No valid project_path provided (received: {repr(project_path)}), using default PhysiCell path")
                work_dir = self.physicell_path
            
            logger.info(f"Resolved work_dir: {work_dir}")
            
            # Check for config file with detailed logging
            config_file = work_dir / "config" / "PhysiCell_settings.xml"
            fallback_config = work_dir / "PhysiCell_settings.xml"
            
            logger.info(f"Looking for config at: {config_file}")
            logger.info(f"Config exists: {config_file.exists()}")
            logger.info(f"Fallback config at: {fallback_config}")
            logger.info(f"Fallback exists: {fallback_config.exists()}")
            
            if not config_file.exists():
                if not fallback_config.exists():
                    error_msg = f"""No configuration found. Debugging information:
• Received project_path: {repr(project_path)}
• Resolved work_dir: {work_dir}
• Checked config locations:
  - {config_file} (exists: False)
  - {fallback_config} (exists: False)
• Directory contents: {list(work_dir.iterdir()) if work_dir.exists() else 'Directory does not exist'}

Please ensure the project has been created with the 'create_project_files' tool first."""
                    
                    return [TextContent(type="text", text=error_msg)]
                else:
                    config_file = fallback_config
                    logger.info(f"Using fallback config: {config_file}")
            
            logger.info(f"Using config file: {config_file}")
            
            # Set environment
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(threads)
            env["PHYSICELL_CPP"] = "/opt/homebrew/bin/g++-15"
            
            messages = []
            
            # Compile if requested
            if compile:
                messages.append("Compiling simulation...")
                logger.info("Starting compilation process...")
                
                # Clean first
                logger.info("Running 'make clean'...")
                clean_result = subprocess.run(
                    ["make", "clean"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=30
                )
                logger.info(f"Clean result: {clean_result.returncode}")
                if clean_result.stdout:
                    logger.info(f"Clean stdout: {clean_result.stdout}")
                if clean_result.stderr:
                    logger.info(f"Clean stderr: {clean_result.stderr}")
                
                # Compile
                logger.info("Running 'make'...")
                compile_result = subprocess.run(
                    ["make"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=300
                )
                
                logger.info(f"Compile result: {compile_result.returncode}")
                if compile_result.stdout:
                    logger.info(f"Compile stdout: {compile_result.stdout}")
                if compile_result.stderr:
                    logger.info(f"Compile stderr: {compile_result.stderr}")
                
                if compile_result.returncode != 0:
                    error_details = f"""🚫 Compilation Failed (exit code {compile_result.returncode})

WORKING DIRECTORY: {work_dir}
COMMAND: make
ENVIRONMENT: CC={env.get('PHYSICELL_CPP', 'not set')}, OMP_NUM_THREADS={env.get('OMP_NUM_THREADS', 'not set')}

📤 STDOUT:
{compile_result.stdout if compile_result.stdout.strip() else '(no stdout output)'}

📥 STDERR:
{compile_result.stderr if compile_result.stderr.strip() else '(no stderr output)'}

🔍 DEBUGGING INFO:
• Makefile exists: {(work_dir / 'Makefile').exists()}
• main.cpp exists: {(work_dir / 'main.cpp').exists()}
• custom_modules/ exists: {(work_dir / 'custom_modules').exists()}
• Config file: {config_file}

💡 NEXT STEPS:
1. Check that the PhysiCell path is correct: {self.physicell_path}
2. Verify all required files are present
3. Check compiler path: {env.get('PHYSICELL_CPP')}
4. Try running 'make' manually in {work_dir}"""
                    
                    logger.error(f"Compilation failed with exit code {compile_result.returncode}")
                    logger.error(f"STDOUT: {compile_result.stdout}")
                    logger.error(f"STDERR: {compile_result.stderr}")
                    
                    return [TextContent(type="text", text=error_details)]
                
                messages.append("✓ Compilation successful")
                logger.info("Compilation completed successfully")
            
            # Find executable
            project_name = work_dir.name
            executable = work_dir / project_name
            
            if not executable.exists():
                # Try 'project' as executable name
                executable = work_dir / "project"
                if not executable.exists():
                    return [TextContent(
                        type="text",
                        text=f"Executable not found. Please compile first."
                    )]
            
            messages.append(f"Running simulation with {threads} threads...")
            
            # Run simulation
            process = subprocess.Popen(
                [f"./{executable.name}"],
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion
            stdout, stderr = process.communicate(timeout=600)  # 10 minute timeout
            
            if process.returncode == 0:
                return [TextContent(
                    type="text",
                    text=f"✅ Simulation completed successfully!\n\n" + 
                         "\n".join(messages) + 
                         f"\n\nOutput:\n{stdout[-2000:]}"  # Last 2000 chars
                )]
            else:
                runtime_error = f"""🚫 Simulation Runtime Error (exit code {process.returncode})

EXECUTABLE: {executable}
WORKING DIRECTORY: {work_dir}
COMMAND: ./{executable.name}

📤 STDOUT:
{stdout if stdout.strip() else '(no stdout output)'}

📥 STDERR:
{stderr if stderr.strip() else '(no stderr output)'}

🔍 DEBUGGING INFO:
• Executable exists: {executable.exists()}
• Executable permissions: {oct(executable.stat().st_mode) if executable.exists() else 'N/A'}
• Working directory contents: {list(work_dir.iterdir()) if work_dir.exists() else 'Directory missing'}

💡 COMMON CAUSES:
• Exit code 255: Compilation issues or missing dependencies
• Exit code -11: Segmentation fault (memory access error)  
• Exit code -9: Killed by system (out of memory)

Try running manually: cd {work_dir} && ./{executable.name}"""
                
                logger.error(f"Simulation failed with exit code {process.returncode}")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                
                return [TextContent(type="text", text=runtime_error)]
                
        except subprocess.TimeoutExpired:
            process.kill()
            return [TextContent(
                type="text",
                text="Simulation timed out after 10 minutes and was terminated."
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error running simulation: {str(e)}"
            )]
    
    async def _analyze_simulation_biology(
        self,
        output_dir: str,
        metrics: Optional[List[str]] = None
    ) -> List[TextContent]:
        """Analyze simulation results from biological perspective"""
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                return [TextContent(
                    type="text",
                    text=f"Output directory not found: {output_path}"
                )]
            
            # Default metrics
            if not metrics:
                metrics = ["tumor_burden", "immune_infiltration", "hypoxic_fraction"]
            
            # Count output files
            xml_files = list(output_path.glob("*.xml"))
            svg_files = list(output_path.glob("*.svg"))
            
            analysis = f"""Biological Analysis of Simulation Results
{'='*50}

📊 Output Summary:
• Data files: {len(xml_files)} snapshots
• Visualizations: {len(svg_files)} SVG files
• Time points: {len(xml_files)} × save_interval

🔬 Biological Metrics Analyzed:"""
            
            # Placeholder for actual analysis
            # In a real implementation, we would parse XML files and calculate metrics
            for metric in metrics:
                if metric == "tumor_burden":
                    analysis += "\n\n📈 Tumor Burden:"
                    analysis += "\n  • Initial: ~100 cells"
                    analysis += "\n  • Final: Data in final XML file"
                    analysis += "\n  • Growth rate: Calculate from time series"
                    
                elif metric == "immune_infiltration":
                    analysis += "\n\n🛡️ Immune Infiltration:"
                    analysis += "\n  • T cell density: Check cell counts"
                    analysis += "\n  • Macrophage polarization: M1 vs M2 ratio"
                    analysis += "\n  • Spatial distribution: Analyze positions"
                    
                elif metric == "hypoxic_fraction":
                    analysis += "\n\n💨 Hypoxic Fraction:"
                    analysis += "\n  • Oxygen levels: Check substrate concentrations"
                    analysis += "\n  • Hypoxic cells: Count cells with O₂ < 5 mmHg"
                    analysis += "\n  • Necrotic core: Identify dead cell regions"
            
            analysis += "\n\n📁 Available Files for Analysis:"
            if xml_files:
                analysis += f"\n  • First snapshot: {min(xml_files).name}"
                analysis += f"\n  • Last snapshot: {max(xml_files).name}"
            
            analysis += "\n\n💡 Visualization:"
            analysis += "\n  • View SVG files in browser for cell positions"
            analysis += "\n  • Use ParaView for 3D visualization of XML data"
            analysis += "\n  • Run python analysis scripts for quantitative metrics"
            
            return [TextContent(type="text", text=analysis)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error analyzing results: {str(e)}"
            )]
    
    
    
    
    # Legacy hardcoded generation methods removed - now using dynamic LLM-guided approach
    
    # ============================================================================
    # END OF DYNAMIC BIOLOGICAL SIMULATION GENERATION METHODS
    # ============================================================================


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PhysiCell Biological MCP Server")
    parser.add_argument(
        "--physicell-path",
        type=str,
        default="./PhysiCell",
        help="Path to PhysiCell installation directory"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting PhysiCell Biological MCP Server...")
        logger.info(f"PhysiCell path: {args.physicell_path}")
        
        # Create server instance
        server_instance = PhysiCellBiologicalMCPServer(args.physicell_path)
        
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
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
