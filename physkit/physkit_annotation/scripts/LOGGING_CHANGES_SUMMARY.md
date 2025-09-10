# Logging Changes Summary

## Overview
I've implemented selective logging functionality to meet your requirements:
1. **Workflow-level logs**: Both console output AND log file
2. **Module-level logs**: ONLY log file, NO console output (treated as DEBUG level)
3. **Log file location**: Same output directory as the workflow

## Changes Made

### 1. PhysKitLogger Enhancements (`physkit_core/logging_config.py`)

**Added new method**: `get_logger_with_selective_handlers()`
- Allows creating loggers with selective console/file output
- File handlers get DEBUG level (all logs)
- Console handlers respect specified level (can be INFO, DEBUG, etc.)
- Maintains backward compatibility

**Key features**:
- `console_output`: Boolean to enable/disable console output
- `log_file`: Optional path for log file
- `level`: Logger level for console output
- File logging always captures all levels (DEBUG and above)

### 2. WorkflowComposer Updates (`physkit_annotation/workflows/workflow_composer.py`)

**Workflow logging setup**:
- Creates `workflow_execution.log` in the output directory
- Console output: INFO level (workflow progress, status updates)
- File output: All levels (complete execution trace)

**Module logging setup**:
- All modules use the same log file (`workflow_execution.log`)
- Console output: NO OUTPUT (completely silent, reduces noise)
- File output: All levels (complete module execution details)

**Automatic configuration**:
- Modules added via `add_module()` or `add_modules()` get automatic logging setup
- Existing modules in `_initialize_module_results()` get logging setup
- All modules share the same log file for easy debugging

### 3. BaseModule Updates (`physkit_annotation/workflows/modules/base_module.py`)

**Enhanced logging**:
- Added logging import
- Modules can use all log levels for detailed logging
- Console output completely disabled for modules (controlled by workflow composer)

## How It Works

### Workflow Execution
```
Console: INFO level - Workflow progress, problem counts, completion status
File:   DEBUG level - Everything including module details, execution flows
```

### Module Execution
```
Console: NO OUTPUT - Completely silent (reduces noise)
File:   DEBUG level - Complete execution details, all log levels
```

### Log File Structure
```
workflow_execution.log
├── Workflow initialization and setup
├── Problem processing progress
├── Module execution details (INFO level)
├── Module debug information (DEBUG level)
├── Error details and stack traces
└── Workflow completion and summary
```

## Benefits

1. **Clean Console Output**: Only workflow-level progress visible
2. **Complete Logging**: All details saved to file for debugging
3. **Centralized Logs**: Single log file per workflow execution
4. **Configurable Levels**: Easy to adjust console vs file verbosity
5. **No Information Loss**: Everything is logged, just filtered for display

## Usage

### Automatic (Recommended)
```python
workflow = WorkflowComposer(
    name="my_workflow",
    output_dir="./output",
    config={"show_progress": True}
)
# Logging is automatically configured
```

### Manual Configuration
```python
# Set environment variable for global logging
export PHYSKIT_LOG_FILE="/path/to/global.log"
export PHYSKIT_LOG_LEVEL="INFO"
```

## File Locations

- **Workflow log**: `{output_dir}/workflow_execution.log`
- **Experiment log**: `{output_dir}/domain_understanding_{timestamp}.log` (existing)
- **Status files**: `{output_dir}/domain_only_annotation_status.json` (existing)

## Testing

Use the provided test script to verify functionality:
```bash
cd physkit/physkit_annotation/scripts/
python3 test_logging.py
```

This will create a test workflow and verify that:
- Log file is created in output directory
- Console shows workflow progress but minimal module details
- Log file contains complete execution information
- Both workflow and module logs are properly captured

## Backward Compatibility

- All existing code continues to work unchanged
- New logging features are opt-in
- Environment variables still work as before
- Existing log files are preserved
