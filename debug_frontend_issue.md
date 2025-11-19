# Frontend Graphing Debug Guide

## The Issue
Server is emitting batch updates correctly, but the graph only shows one point.

## Debugging Steps

### Step 1: Check Browser Console
Open your browser DevTools (Press F12), go to the Console tab, and look for:

**Expected messages:**
```
✓ Socket connected!
⚙️ Batch 1/50 (epoch 1) - Loss: 0.5234, step 1
⚙️ Batch 10/50 (epoch 1) - Loss: 0.4891, step 10
⚙️ Batch 20/50 (epoch 1) - Loss: 0.4567, step 20
```

**Questions to answer:**
1. ✅ Do you see "✓ Socket connected!" ?
2. ✅ Do you see the "⚙️ Batch X/Y..." messages for MULTIPLE batches?
3. ❌ Are there any JavaScript errors in red?

### Step 2: Check if Data is Being Added

In the browser console, type:
```javascript
trainingData
```

**Expected output:**
```javascript
{
  epochs: [1, 10, 20, 30, ...],
  loss: [0.5234, 0.4891, 0.4567, ...],
  accuracy: [0.45, 0.52, 0.58, ...],
  posSim: [0.7, 0.72, 0.74, ...],
  negSim: [0.3, 0.28, 0.26, ...]
}
```

**If all arrays are empty or have only 1 element**, the batch_update handler is not working.

### Step 3: Manually Test the Chart

In the browser console, try manually adding data:
```javascript
// Add test data
trainingData.epochs.push(1, 2, 3, 4, 5);
trainingData.loss.push(0.5, 0.4, 0.3, 0.2, 0.1);
trainingData.accuracy.push(0.5, 0.6, 0.7, 0.8, 0.9);
trainingData.posSim.push(0.7, 0.71, 0.72, 0.73, 0.74);
trainingData.negSim.push(0.3, 0.29, 0.28, 0.27, 0.26);

// Manually update charts
updateCharts();
```

**If the chart updates**, the issue is with the batch_update event handler.
**If the chart doesn't update**, the issue is with the Plotly chart rendering.

## Common Frontend Issues

### Issue #1: initial_state Event Resetting Data
**Location:** dashboard.html line 732

The `initial_state` event handler resets the chart data:
```javascript
socket.on('initial_state', (state) => {
    // This CLEARS all chart data!
    trainingData.epochs = [];
    trainingData.loss = [];
    // ...
});
```

**Problem:** If `initial_state` is emitted AFTER training starts, it will clear the data.

**Check server logs:** Look for "CLIENT CONNECTED" and "Initial state sent" messages.

### Issue #2: Data Type Mismatch
**Location:** dashboard.html line 772-777

The code checks if similarity values are numbers:
```javascript
if (typeof data.avg_positive_sim === 'number') {
    trainingData.posSim.push(data.avg_positive_sim);
}
```

**Problem:** If the data isn't a number, posSim and negSim arrays will have different lengths than epochs/loss/accuracy arrays.

**Solution:** This causes Plotly to fail silently.

### Issue #3: Plotly.react() Failing
**Location:** dashboard.html line 486 and 512

The updateCharts() function uses `Plotly.react()`:
```javascript
Plotly.react('loss-accuracy-chart', [lossTrace, accuracyTrace], {...});
```

**Problem:** If there's a configuration error or data format issue, Plotly might fail silently.

**Check:** Look for errors in the browser console.

## Most Likely Causes

Based on the symptoms (one point, then nothing), the most likely issues are:

1. **initial_state event being called after training starts** - Clears the data
2. **JavaScript error in batch_update handler** - Breaks after first update
3. **Plotly configuration error** - Chart not updating properly

## Next Steps

1. **Check browser console** during training
2. **Look for the "⚙️ Batch..." messages**
3. **Check for any red error messages**
4. **Inspect the trainingData object** to see if arrays are growing

Report back what you see!
