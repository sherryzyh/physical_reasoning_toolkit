# Progress Bar Functionality in Workflow Composer

The `WorkflowComposer` now includes progress tracking functionality to monitor the processing of large datasets.

## Features

- **Simple Progress Bar**: Shows overall dataset processing progress
- **Real-time Statistics**: Displays current problem ID, success count, and failure count
- **Configurable**: Can be enabled/disabled via configuration
- **Graceful Fallback**: Works with or without `tqdm` library

## Configuration

### Enable Progress Bar (Default)
```python
workflow = WorkflowComposer(
    name="my_workflow",
    output_dir="./output",
    config={"show_progress": True}  # Default behavior
)
```

### Disable Progress Bar
```python
workflow = WorkflowComposer(
    name="my_workflow",
    output_dir="./output",
    config={"show_progress": False}
)
```

## What You'll See

When progress bars are enabled, you'll see:

```
Processing problems: 45%|████▌     | 45/100 [00:30<00:37, 1.47problem/s] Current: problem_045, Success: 43, Failed: 2
```

The progress bar shows:
- **Percentage**: Overall completion (45%)
- **Visual Bar**: Visual representation of progress
- **Count**: Current problem / Total problems (45/100)
- **Time**: Elapsed time and estimated time remaining
- **Rate**: Problems processed per second
- **Current**: ID of the problem being processed
- **Success/Failed**: Running counts of successful and failed problems

## Dependencies

The progress bar requires the `tqdm` library. If it's not available, the workflow will still work but without visual progress bars.

### Install tqdm
```bash
pip install tqdm
```

## Logging

Even without progress bars, the workflow logs progress information:
```
INFO: Processing problem 45/100: problem_045
```

## Example Usage

```python
from physkit_annotation.workflows.workflow_composer import WorkflowComposer

# Create workflow with progress bar enabled
workflow = WorkflowComposer(
    name="domain_annotation_workflow",
    output_dir="./output",
    config={"show_progress": True}
)

# Add your modules
workflow.add_module(domain_assessment_module)
workflow.add_module(annotation_module)

# Run with progress tracking
results = workflow.run(dataset)
```

## Benefits

1. **Visibility**: Know exactly how many problems remain to be processed
2. **Time Estimation**: Get realistic estimates of completion time
3. **Monitoring**: Track success/failure rates in real-time
4. **Debugging**: Identify which specific problem is being processed
5. **User Experience**: Better feedback during long-running workflows

## Performance Impact

The progress bar adds minimal overhead:
- Only updates when problems are processed
- No impact on the actual processing speed
- Memory usage remains the same
- Can be disabled for maximum performance if needed
