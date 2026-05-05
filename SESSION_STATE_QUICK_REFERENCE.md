# 💡 Streamlit Session State - Quick Reference & Best Practices

## The Problem & Solution in 30 Seconds

### ❌ Problem
```python
model = None
if st.button("Train"):
    model = train()
    show_results(model)
# User switches page → model = None again! ❌
```

### ✅ Solution
```python
if "model" not in st.session_state:
    st.session_state.model = None

with st.form("train"):
    if st.form_submit_button("Train"):
        st.session_state.model = train()

if st.session_state.model:
    show_results(st.session_state.model)  # Persists across reruns! ✓
```

---

## When to Use Session State

### ✅ USE session_state for:
- Machine learning models
- Trained parameters/weights
- User progress/history
- Computation results
- File uploads (metadata)
- Form submissions
- Multi-page workflows
- Any data that should survive reruns

### ❌ DON'T use session_state for:
- Widget values (st.slider, st.input) - automagically handled
- Rendering parameters
- Temporary loop variables
- Debug/logging info

---

## Session State Patterns

### Pattern 1: Check and Initialize
```python
if "key" not in st.session_state:
    st.session_state.key = default_value
```

### Pattern 2: Safe Set with Default
```python
st.session_state.setdefault("key", default_value)
```

### Pattern 3: Using Forms for Control
```python
with st.form("my_form"):
    input_val = st.number_input("Number:")
    if st.form_submit_button("Submit"):
        st.session_state.result = compute(input_val)

if st.session_state.get("result"):
    st.write(st.session_state.result)
```

### Pattern 4: Conditional Display
```python
if st.session_state.get("is_trained"):
    display_model_info()
else:
    st.info("Train a model first")
```

### Pattern 5: Reset/Clear
```python
if st.button("Reset"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()
```

---

## Access Methods

### Method 1: Dot Notation
```python
st.session_state.key = value     # Set
value = st.session_state.key     # Get
```

### Method 2: Bracket Notation (Safer)
```python
st.session_state["key"] = value  # Set
value = st.session_state["key"]  # Get
```

### Method 3: get() with Default
```python
value = st.session_state.get("key", default)  # Safe get
```

### Method 4: Check Existence
```python
if "key" in st.session_state:    # Check
if "key" not in st.session_state:  # Negative check
```

---

## Real-World Example: ML Pipeline

```python
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize session state
@st.cache_resource
def init_state():
    if "model" not in st.session_state:
        st.session_state.model = None
    if "scaler" not in st.session_state:
        st.session_state.scaler = None
    if "trained" not in st.session_state:
        st.session_state.trained = False

init_state()

# Page 1: Train Model
if page == "Train":
    st.header("Train Model")
    
    with st.form("training_form"):
        n_trees = st.slider("Number of trees:", 10, 200, 100)
        submitted = st.form_submit_button("Train")
    
    if submitted:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = RandomForestClassifier(n_estimators=n_trees)
        model.fit(X_train, y_train)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.accuracy = model.score(X_test, y_test)
        st.session_state.trained = True
        st.success("Model trained!")
    
    # Display results after training
    if st.session_state.trained:
        st.metric("Accuracy", f"{st.session_state.accuracy:.4f}")

# Page 2: Predict
if page == "Predict":
    st.header("Make Prediction")
    
    if not st.session_state.trained:
        st.warning("Train a model first!")
    else:
        with st.form("predict_form"):
            input_data = st.text_input("Enter features:")
            if st.form_submit_button("Predict"):
                prediction = st.session_state.model.predict([features])
                st.write(f"Prediction: {prediction}")

# Page 3: Analysis
if page == "Analysis":
    st.header("Model Analysis")
    
    if st.session_state.trained:
        st.metric("Accuracy", f"{st.session_state.accuracy:.4f}")
        st.write(st.session_state.model)
    else:
        st.info("No model trained yet")
```

---

## Common Pitfalls & Solutions

### ❌ Pitfall 1: Forgetting to Initialize
```python
# This will crash if key doesn't exist
value = st.session_state.key  # KeyError on first run!

# ✅ Solution: Check first
value = st.session_state.get("key", default_value)
```

### ❌ Pitfall 2: Storing in Local Variable
```python
model = st.session_state.model  # Creates a reference
model.fit(X, y)                  # Modifies the original!
# Session state is updated too

# ✅ Solution: Store directly
st.session_state.model.fit(X, y)
```

### ❌ Pitfall 3: Expecting Widget Values to Persist
```python
value = st.slider("Value:")  # Don't save this in session_state
st.session_state.slider_value = value  # Unnecessary!

# ✅ Solution: Widgets already persist
if some_condition:
    st.slider("Value:")  # Value persists automatically
```

### ❌ Pitfall 4: Not Using st.form
```python
epochs = st.slider("Epochs:")
batch_size = st.slider("Batch Size:")
# Rerun on every adjustment! Slow!

# ✅ Solution: Use st.form
with st.form("config"):
    epochs = st.slider("Epochs:")
    batch_size = st.slider("Batch Size:")
    if st.form_submit_button("Train"):
        # Only rerun when submit clicked
        train(epochs, batch_size)
```

### ❌ Pitfall 5: Mutating Objects
```python
st.session_state.data = []
st.session_state.data.append(item)  # Works but prefer explicit assignment

# ✓ Better: Be explicit
data = st.session_state.get("data", [])
data.append(item)
st.session_state.data = data  # Explicit storage
```

---

## Debugging Session State

### View All Data
```python
with st.expander("Debug: Session State"):
    st.write(st.session_state)
    st.write(dict(st.session_state))  # As dictionary
```

### View Specific Keys
```python
st.write(f"Model trained: {st.session_state.get('trained', False)}")
st.write(f"Model type: {type(st.session_state.get('model'))}")
st.write(f"Keys: {list(st.session_state.keys())}")
```

### Clear and Reset
```python
if st.button("Clear Session State"):
    st.session_state.clear()
    st.rerun()

if st.button("Reset Model"):
    st.session_state.model = None
    st.session_state.trained = False
    st.rerun()
```

### Monitor Changes
```python
# Log changes
import json

previous_state = st.session_state.get("_prev_state", {})
current_state = dict(st.session_state)

changed = {k: v for k, v in current_state.items() if k not in previous_state or previous_state[k] != v}
if changed:
    st.write(f"Changed: {changed}")

st.session_state._prev_state = current_state
```

---

## Performance Tips

### Tip 1: Cache Expensive Operations
```python
@st.cache_resource
def load_model():
    return train_expensive_model()

model = load_model()
st.session_state.model = model
```

### Tip 2: Use st.form for Batch Updates
```python
# ❌ Bad: Rerun on every input
epochs = st.slider("Epochs:", 1, 100)
batch = st.slider("Batch:", 1, 128)
# Reruns TWICE while adjusting

# ✅ Good: Single rerun
with st.form("config"):
    epochs = st.slider("Epochs:", 1, 100)
    batch = st.slider("Batch:", 1, 128)
    if st.form_submit_button("Apply"):
        pass  # Only rerun when submit clicked
```

### Tip 3: Store Only What's Needed
```python
# ❌ Don't store entire DataFrame if you only need a metric
st.session_state.full_data = huge_dataframe

# ✅ Store the computed metric
st.session_state.accuracy = 0.92
st.session_state.metadata = {"rows": 1000, "cols": 50}
```

### Tip 4: Use Lazy Evaluation
```python
# Compute only when needed
if st.session_state.get("need_results"):
    results = expensive_compute()
    st.session_state.results = results
```

---

## Architecture Pattern: Multi-Page App with Session State

```python
# app.py
import streamlit as st
from pages import train, predict, analyze

# Initialize
def init_app():
    if "model" not in st.session_state:
        st.session_state.model = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = None

init_app()

# Navigation
page = st.sidebar.radio("Choose:", 
    ["🏠 Home", "🏋️ Train", "🔮 Predict", "📊 Analyze"])

if page == "🏠 Home":
    st.write("Welcome!")
elif page == "🏋️ Train":
    train.show()  # Stores to st.session_state
elif page == "🔮 Predict":
    predict.show(st.session_state.model)  # Uses stored model
elif page == "📊 Analyze":
    analyze.show(st.session_state.metrics)  # Uses stored metrics
```

---

## Comparison: Before and After

### ❌ BEFORE (Loses Data on Rerun)
```python
def main():
    model = None
    metrics = {}
    
    if st.button("Train"):
        model = train()
        metrics = evaluate(model)
        st.write(metrics)  # Only shown if button pressed
    
    # After rerun: model and metrics are None!

main()
```

### ✅ AFTER (Persistent)
```python
def init():
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("metrics", {})
    st.session_state.setdefault("trained", False)

init()

with st.form("train"):
    if st.form_submit_button("Train"):
        st.session_state.model = train()
        st.session_state.metrics = evaluate(st.session_state.model)
        st.session_state.trained = True

# Display persists across reruns!
if st.session_state.trained:
    st.write(st.session_state.metrics)
```

---

## Decision Tree: Should I Use Session State?

```
Does the data need to
survive across reruns?
    ↓
    ├─ YES → Use st.session_state
    │    ├─ Is it a model/weights? → Store it
    │    ├─ Is it a computation result? → Store it
    │    └─ Is it user progress? → Store it
    │
    └─ NO → Don't use session_state
         ├─ Is it a widget value? → Let Streamlit handle it
         ├─ Is it temporary? → Use local variable
         └─ Is it just for display? → Use st.write directly
```

---

## Resources

### Official Streamlit Docs
- [Session State Documentation](https://docs.streamlit.io/develop/api-reference/session-state)
- [Session State API Reference](https://docs.streamlit.io/develop/api-reference/session-state)
- [Multi-Page Apps](https://docs.streamlit.io/develop/concepts/multipage-apps/overview)

### Best Practices
✅ Always initialize before use  
✅ Use st.form for user input  
✅ Store computation results  
✅ Display data outside logic blocks  
✅ Check status flags  
✅ Use clear naming conventions  
✅ Document your session_state structure  

---

## Summary Table

| Need | Solution | Example |
|------|----------|---------|
| **Persist model** | st.session_state | `st.session_state.model = trained_model` |
| **Check if done** | Boolean flag | `st.session_state.is_trained` |
| **Condition display** | if statement | `if st.session_state.is_trained: show()` |
| **Prevent reruns** | st.form | `with st.form("name"): ...` |
| **Access safely** | .get() method | `st.session_state.get("key", default)` |
| **Initialize all** | Function at start | `def init_state(): ...` then `init_state()` |
| **Debug data** | Expander + write | `with st.expander("Debug"): st.write(dict(st.session_state))` |

---

## Final Checklist

✅ **App setup:**
- [ ] Initialize session_state on every run
- [ ] Use clear, descriptive key names
- [ ] Document session_state structure

✅ **Data storage:**
- [ ] Store models/results in session_state
- [ ] Set status flag when ready
- [ ] Don't store unnecessary data

✅ **UI implementation:**
- [ ] Use st.form for user input
- [ ] Display results outside form/button blocks
- [ ] Check status before using stored data

✅ **Error handling:**
- [ ] Use .get() for safe access
- [ ] Validate before using stored data
- [ ] Handle missing data gracefully

✅ **Testing:**
- [ ] Test page switching
- [ ] Test multiple sessions
- [ ] Debug with st.write(st.session_state)

---

**Master session_state and your Streamlit apps will be bulletproof! 🚀**
