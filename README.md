# regime
The `regime` library offers a precise framework to outline workflows consisting of classes, functions, and resources. The `Regime` class uses `Process` and `Resource` objects to delineate the flow of algorithms and input or output byproducts. `Process` objects, if inheriting from `HyperparameterMeta`, can explicitly "tag" hyperparameters by using the `hyperparameter` decorator; this allows for the clear separation of hyperparameters such as those found in experiments (e.g., _alpha_, _beta_) and ordinary arguments (e.g., _dataset_). 

Special features available through `regime`:

1. *Hyperparameter Recognition*: We can always automatically determine what are the hyperparameters from a `Process` signature. In doing so, this allows us to know which arguments we can safely explore other values.
2. *Hyperparameter Validation*: Keeping up with hyperparameters for many processes can quickly become cumbersome - especially in complex workflows. To address this - `Regime` determines what hyperparameters must be defined to use the required `Process` objects, and checks that these are provided via a `dict` instance. This `dict` follows a hierarchical structure that comes directly from Python modules' paths to ensure that hyperparameters remain unique and their purpose known (i.e., they are nested according to the exact location they are found).
3. *Hyperparameter Logging*: Often, hyperparameters require _fine-tuning_, and after an experiment is performed - if the results are ideal, we wish to store these values for later reuse. The hyperparameters used for a `Regime` object can easily be exported as .yaml files.
4. *Workflow Visualization*: Due to `Regime`'s backend graph to implement the flow of data between `Process` instances, your program's workflow is readily able to be visualized by using the `igraph` library! This allows you to dynamically create diagrams showcasing how your program's functions, classes, resources, etc. all interact with each other, and can serve as a form of additional real-time documentation (e.g., PDF file).
5. *Process Inspection*: The `Regime` class inherits from the features implemented in the `rough-theory` library. This enables `Regime` instances to leverage operations analyzing discernibility for complex analysis of workflows.

Incorporating `regime` into your code is straightforward and requires minimal edits! 
