# PhysKit Cookbooks

A collection of practical examples and tutorials for using PhysKit.

## 🚀 Quick Start

### Prerequisites
- PhysKit packages installed (`physkit_datasets`, `physkit_annotation`, `physkit_evaluation`)
- OpenAI API key set (for annotation and evaluation cookbooks)
- Dataset files available in your data directory

### Basic Usage
```bash
# Load and explore datasets
python 01_load_dataset.py ugphysics
python 01_load_dataset.py phybench
python 01_load_dataset.py physreason

# Run annotation workflows
python 02_automated_annotation.py


# Demo environment variables and workflow composition
python 03_environment_variables.py
python 04_workflow_composition_demo.py


# Test enhanced evaluation toolkit (25 comprehensive test scenarios)
python 05_answer_comparison_demo.py
```

## 📚 Available Cookbooks

### 1. **Dataset Loading & Exploration** (`01_load_dataset.py`)
**Purpose**: Load, explore, and test any PhysKit dataset

**Features**:
- ✅ **Flexible**: Test any dataset by name (ugphysics, phybench, physreason, etc.)
- ✅ **Configurable**: Customize variant, split, sample size, and data directory
- ✅ **Comprehensive**: Shows dataset info, structure, sample problems, and statistics
- ✅ **Debug-friendly**: Detailed error messages and debugging output

**Usage Examples**:
```bash
# Test UGPhysics dataset (default)
python 01_load_dataset.py

# Test specific dataset
python 01_load_dataset.py phybench

# Test with custom parameters
python 01_load_dataset.py ugphysics --variant full --split test --sample-size 5

# Test with custom data directory
python 01_load_dataset.py physreason --data-dir ~/my_data
```

**Command Line Options**:
- `dataset_name`: Name of dataset to test (default: ugphysics)
- `--data-dir`: Data directory path (default: auto-detect)
- `--variant`: Dataset variant (default: full)
- `--split`: Dataset split (default: test)
- `--sample-size`: Number of problems to sample (default: all)

**Output**:
- Dataset information and statistics
- Sample problem structures
- Different loading options demonstration
- Saved files in `showcase_output/dataset_exploration/`

### 2. **Automated Annotation Workflow** (`02_automated_annotation.py`)
**Purpose**: Run unsupervised LLM-based annotation on physics problems

**Features**:
- ✅ **Sequential Pipeline**: Domain → Theorem → Variable → Final Answer annotation
- ✅ **LLM Integration**: Uses OpenAI models for automated annotation
- ✅ **Configurable**: Sample sizes and model selection
- ✅ **Comprehensive Results**: Saves individual problem annotations and workflow statistics
- ✅ **Error Handling**: Graceful handling of failed annotations with detailed error reporting

**Usage**:
```bash
python 02_automated_annotation.py
```

**Prerequisites**:
- `OPENAI_API_KEY` environment variable set
- UGPhysics dataset available

**Output**:
- Individual problem annotation files in `annotation/` directory
- Workflow statistics and summary
- Detailed logging of each annotation step

### 3. **Environment Variables Demo** (`03_environment_variables.py`)
**Purpose**: Demonstrate environment variable configuration and priority

**Features**:
- Shows how to configure PhysKit with environment variables
- Demonstrates environment variable priority
- Configures API keys and settings
- Saves configuration examples

**Usage**:
```bash
python 03_environment_variables.py
```

### 4. **Workflow Composition Demo** (`04_workflow_composition_demo.py`)
**Purpose**: Demonstrate how to compose custom annotation workflows using WorkflowComposer

**Features**:
- ✅ **Custom Workflow Creation**: Build workflows by combining individual modules
- ✅ **Module Composition**: Add, remove, and chain workflow modules
- ✅ **Flexible Configuration**: Customize workflow parameters and settings
- ✅ **Result Analysis**: Comprehensive workflow statistics and data flow analysis
- ✅ **Status Monitoring**: Real-time workflow status and control capabilities

**Usage**:
```bash
python 04_workflow_composition_demo.py
```

**Prerequisites**:
- `OPENAI_API_KEY` environment variable set
- PHYBench dataset available

**Key Concepts Demonstrated**:
- WorkflowComposer for orchestration
- Module composition and chaining
- Result aggregation and statistics
- Output management and file organization

### 5. **Enhanced Evaluation Toolkit Demo** (`05_answer_comparison_demo.py`)
**Purpose**: Comprehensive demonstration of PhysKit's advanced evaluation capabilities across different answer types

**Features**:
- ✅ **Multi-type comparison**: Symbolic expressions, numerical values, textual descriptions, and multiple choice options
- ✅ **Advanced symbolic parsing**: Handles complex LaTeX, equations vs expressions, mathematical equivalence
- ✅ **Smart numerical comparison**: Significant figure-based comparison, unit conversions, special cases (infinity, NaN, zero)
- ✅ **Semantic textual analysis**: LLM-powered comparison for different phrasings and explanations
- ✅ **Intelligent option comparison**: Case-insensitive, order-independent multiple choice answer comparison
- ✅ **Comprehensive test scenarios**: 25 diverse physics problems covering various edge cases
- ✅ **Detailed analysis**: Per-type accuracy breakdown, comparison method details, error analysis

**Test Scenarios**:
- **Symbolic (5 problems)**: Complex velocity functions, Newton's laws, Einstein's E=mc², integral equations
- **Numerical (8 problems)**: Unit conversions (km/h ↔ m/s, °F ↔ °C, g ↔ kg), significant figures, special values
- **Textual (5 problems)**: Physics explanations with different phrasings and terminology
- **Option (7 problems)**: Single choice, multiple choice, case-insensitive, order-independent, different separators

**Usage**:
```bash
python 05_answer_comparison_demo.py
```

**Prerequisites**:
- `physkit_evaluation` package installed
- `physkit` package installed
- OpenAI API access for LLM-based comparisons

**Sample Output**:
```
🎯 Key Features Demonstrated:
  • Symbolic: Handles equations vs expressions, complex LaTeX parsing
  • Numerical: Unit conversions, significant figures, special cases (inf, NaN)
  • Textual: Semantic similarity using LLM comparison
  • Option: Case-insensitive, order-independent multiple choice comparison
  • Comprehensive error handling and detailed comparison results

📈 Accuracy Breakdown by Answer Type
Symbolic:   80.00% (4/5)
Numerical:  75.00% (6/8)  
Textual:    100.00% (5/5)
Option:     100.00% (7/7)
```

**Advanced Capabilities Showcased**:
- **LaTeX Processing**: PhyBench preprocessing with `posify`, `time_simplify`, equation parsing
- **Unit Intelligence**: Dimensional analysis, automatic conversion factors, compatibility checking
- **Significant Figures**: Precision-aware comparison without fixed tolerance
- **LLM Integration**: GPT-4o for semantic comparison of units and explanations
- **Option Intelligence**: Smart multiple choice comparison with normalization, case-insensitive matching, order independence





## 📥 Download Datasets

Before running the cookbooks, you need to download the required datasets. PhysKit currently supports three main datasets:

### **Supported Datasets**
- **UGPhysics**: Undergraduate physics problems across 13 domains
- **SeePhys**: Visual physics problems with images
- **PHYBench**: Physics benchmark dataset
- **PhysReason**: Physics reasoning problems with step-by-step solutions
- **JEEBench**: Challenging problems from IIT JEE-Advanced examination across Physics, Chemistry, and Mathematics

### **Dataset Structure**
Your datasets should be organized as follows:

```
~/data/                          # Your dataset root directory
├── ugphysics/                   # UGPhysics dataset
│   ├── AtomicPhysics/
│   │   ├── en.jsonl            # English problems
│   │   └── zh.jsonl            # Chinese problems
│   ├── ClassicalMechanics/
│   │   ├── en.jsonl
│   │   └── zh.jsonl
│   ├── QuantumMechanics/
│   │   ├── en.jsonl
│   │   └── zh.jsonl
│   └── ...                     # 10 more domains
├── SeePhys/                     # SeePhys dataset
│   ├── physreason_format/      # Converted to PhysReason format
│   │   ├── 1900.json
│   │   ├── 1901.json
│   │   └── ...                 # 2000+ problem files
│   ├── images/                 # Problem images
│   ├── seephys_train.csv      # Training data
│   └── seephys_train_mini.csv # Mini training data
└── PHYBench/                   # PHYBench dataset
    ├── PHYBench-fullques_v1.json
    ├── PHYBench-onlyques_v1.json
    └── PHYBench-questions_v1.json

```

### **Download Instructions**

#### **Option 1: Manual Download (Recommended)**
1. **UGPhysics**: Download from [Hugging Face Dataset](https://huggingface.co/datasets/UGPhysics/ugphysics)
2. **SeePhys**: Download from [SeePhys repository](https://github.com/AI4Phys/SeePhys?tab=readme-ov-file)
3. **PHYBench**: Download from [Hugging Face Dataset](https://huggingface.co/datasets/Eureka-Lab/PHYBench)

#### **Option 2: Custom Data Directory**
If you want to use a different directory structure:
```bash
# Set custom data directory when running cookbooks
python 01_load_dataset.py ugphysics --data-dir /path/to/your/datasets
```

### **Verification**
After downloading, verify your datasets:
```bash
# Test dataset loading
python 01_load_dataset.py ugphysics
python 01_load_dataset.py seephys
python 01_load_dataset.py phybench
```

## 🔧 Setup & Configuration

### Environment Setup
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Or create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Quick Environment Check
```bash
# Run the setup script to check your environment
python setup_cookbooks.py
```

This will verify:
- Python version compatibility
- Required PhysKit packages
- OpenAI API key configuration
- Dataset availability

### Data Directory Structure
```
../data/
├── ugphysics/           # UGPhysics dataset
│   ├── mini/
│   │   ├── AtomicPhysics/
│   │   │   └── en.jsonl
│   │   └── ...
│   └── full/
├── phybench/            # PHYBench dataset
├── physreason/          # PhysReason dataset
└── ...                  # Other datasets
```

### Custom Data Paths
```bash
# Use custom data directory
python 01_load_dataset.py ugphysics --data-dir /path/to/your/data

# Test different variants
python 01_load_dataset.py ugphysics --variant full

# Test different splits
python 01_load_dataset.py phybench --split validation
```

## 🚨 Troubleshooting

### Common Issues

**1. Dataset Not Found**
```bash
❌ Dataset 'unknown_dataset' not found!
Available datasets: ugphysics, phybench, seephys
```
**Solution**: Use one of the available dataset names

**2. Data Directory Not Found**
```bash
❌ Data directory not found: ../data
```
**Solution 1**: (recommended) Download datasets using the instructions in the [Download Datasets](#-download-datasets) section above
**Solution 2**: Create a "data/" folder under your home directory and create sub-folders with dataset names
**Solution 3**: Set correct data directory with `--data-dir` flag

**3. OpenAI API Key Missing**
```bash
❌ OPENAI_API_KEY environment variable not set
```
**Solution**: Set `OPENAI_API_KEY` environment variable or create an `.env` file

**4. Evaluation Toolkit Import Error**
```bash
ModuleNotFoundError: No module named 'physkit_evaluation'
```
**Solution**: Ensure the `physkit_evaluation` package is properly installed and the import paths are correctly configured


## 📁 Showcase Output Structure

```
showcase_output/
├── dataset_exploration/
│   ├── ugphysics_summary.txt
│   ├── ugphysics_sample_problems.json
│   ├── phybench_summary.txt
│   └── ...
├── automated_annotation/
│   ├── annotation/
│   │   ├── problem_1.json
│   │   └── problem_2.json
│   └── annotation_workflow.log
├── workflow_composition_demo/
│   ├── domain_assessment_demo_workflow_statistics.json
│   └── domain_assessment_demo_results.json
└── evaluation_results/
    ├── symbolic_comparison_details.json
    ├── numerical_comparison_details.json
    ├── textual_comparison_details.json
    └── comprehensive_evaluation_report.txt
```

---

**Happy cooking with PhysKit! 🧪⚡**
