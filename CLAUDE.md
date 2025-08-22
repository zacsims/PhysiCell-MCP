# PhysiCell MCP Dynamic Simulation Generation Plan

## Vision
Transform the PhysiCell MCP from a generic template generator into a **biological simulation consultant** that understands biology and generates complete, scientifically accurate PhysiCell models dynamically based on user descriptions.

## Core Problem
Current approach creates generic templates with basic cell types, but users want domain-specific, biologically realistic simulations that are generated dynamically based on their biological scenario descriptions.

## Solution Architecture

### 1. Biological Knowledge Engine

The MCP should understand and reason about:

#### Cell Types
- **Tumor cells**: proliferation rates, oxygen dependencies, contact inhibition
- **Immune cells**: 
  - T cells (CD8+, CD4+, regulatory)
  - Macrophages (M1 pro-inflammatory, M2 anti-inflammatory)
  - NK cells, B cells
- **Stromal cells**:
  - Fibroblasts (CAFs)
  - Endothelial cells
  - Pericytes

#### Microenvironment Factors
- **Nutrients**: oxygen, glucose, lactate
- **Signaling molecules**: growth factors (EGF, VEGF), cytokines (TNF-α, IL-6)
- **Therapeutic agents**: checkpoint inhibitors, chemotherapy
- **Physical factors**: pressure, pH, temperature

#### Biological Processes
- Cell proliferation and death (apoptosis, necrosis)
- Migration and chemotaxis
- Cell-cell interactions and signaling
- Differentiation and phenotype switching
- Angiogenesis and vascular remodeling

### 2. Smart Parameter Generation System

Automatically determine biologically realistic parameters from context:

#### Cell Parameters
```python
def generate_tumor_cell_parameters(context):
    return {
        "proliferation_rate": 7.2e-4,  # ~24 hour doubling time
        "apoptosis_rate": 5.3e-5,
        "necrosis_oxygen_threshold": 3.75,  # mmHg
        "volume": 2494,  # cubic microns
        "mechanics": {
            "adhesion": 0.4,
            "repulsion": 10.0,
            "migration_speed": 0.1
        }
    }
```

#### Substrate Properties
```python
def generate_oxygen_parameters():
    return {
        "diffusion_coefficient": 100000,  # micron^2/min
        "decay_rate": 0.1,  # 1/min
        "initial_condition": 38,  # mmHg
        "boundary_condition": 38  # mmHg
    }
```

#### Interaction Rules
Generate biologically meaningful cell-substrate and cell-cell interactions:
- Oxygen-dependent proliferation
- Inflammatory factor responses
- Contact-dependent behaviors
- Chemotactic migration

### 3. Enhanced MCP Tools

#### create_biological_simulation
```python
async def create_biological_simulation(description: str, parameters: dict):
    """
    Dynamically generate a complete PhysiCell simulation from biological description
    
    Example:
        description: "tumor microenvironment with immune infiltration"
        
    Generates:
        - Tumor cells with oxygen-dependent behavior
        - Macrophages with inflammatory responses
        - T cells with tumor-killing capability
        - Realistic microenvironment (oxygen, cytokines)
        - Complete interaction rules
    """
```

#### analyze_biological_scenario
```python
async def analyze_biological_scenario(description: str):
    """
    Parse user description to identify:
    - Key cell types needed
    - Relevant substrates
    - Important biological processes
    - Spatial organization requirements
    """
```

#### generate_cell_rules
```python
async def generate_cell_rules(cell_types: list, context: str):
    """
    Generate CSV rules file with biologically realistic interactions
    Based on known biological relationships
    """
```

### 4. Dynamic Generation Pipeline

1. **Parse biological intent** from user description
2. **Identify key components** (cells, substrates, processes)
3. **Generate realistic parameters** for each component
4. **Create interaction rules** based on biological knowledge
5. **Set up spatial initialization** (tissue architecture)
6. **Generate complete PhysiCell project** structure
7. **Validate biological consistency**

### 5. Example Workflows

#### Example 1: Tumor Hypoxia
**User:** "Create a simulation of tumor hypoxia response"

**MCP Process:**
1. Identifies: tumor cells, oxygen gradients, hypoxic responses
2. Generates:
   - Tumor cells with oxygen-dependent proliferation
   - Oxygen substrate with diffusion limitations
   - Hypoxia-induced cell death
   - VEGF secretion under hypoxia
3. Creates complete simulation

#### Example 2: Cancer Immunotherapy
**User:** "Model checkpoint inhibitor therapy in pancreatic cancer"

**MCP Process:**
1. Identifies: PDAC cells, T cells, checkpoint molecules
2. Generates:
   - PD-L1+ tumor cells
   - PD-1+ T cells with exhaustion
   - Checkpoint inhibitor drug dynamics
   - Immune activation/suppression rules
3. Creates therapy simulation

### 6. Implementation Priorities

1. **Phase 1: Knowledge Base**
   - Extract patterns from existing user_projects
   - Build biological parameter database
   - Create rule generation templates

2. **Phase 2: Dynamic Generation**
   - Implement biological scenario parser
   - Build parameter generation functions
   - Create rule generation system

3. **Phase 3: Integration**
   - Update MCP tools
   - Add validation systems
   - Create result analysis tools

4. **Phase 4: Testing**
   - Test with various biological scenarios
   - Validate against published models
   - Refine parameters based on results

## Key Differentiators

1. **No pre-defined templates** - Everything generated dynamically
2. **Biologically informed** - Parameters based on real biological data
3. **Context-aware** - Understands relationships between components
4. **Complete simulations** - Not just templates, but working models
5. **Scientifically valid** - Generates publishable-quality simulations

## Success Metrics

- User can describe any biological scenario and get working simulation
- Generated parameters match published biological values
- Simulations produce biologically realistic behaviors
- Zero manual coding required for standard scenarios

## Next Steps

1. Build biological knowledge database from user_projects analysis
2. Implement parameter generation functions
3. Create dynamic rule generation system
4. Update MCP server with new tools
5. Test with diverse biological scenarios