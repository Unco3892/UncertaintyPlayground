# define the class for the variational layer
k_set_floatx("float64")
bt <- import("builtins")
# RBFKernelFn <- reticulate::PyClass(
#     "KernelFn",
#     inherit = tensorflow::tf$keras$layers$Layer,
#     list(
#         `__init__` = function(self, dtype = NULL, trainable = TRUE, ...) {
#             super()$`__init__`(trainable = trainable, ...)

#             self$`_amplitude` <- self$add_variable(
#                 initializer = initializer_constant(value = 1.0),
#                 dtype = dtype,
#                 name = "amplitude"
#             )
#             self$`_length_scale` <- self$add_variable(
#                 initializer = initializer_constant(value = 1.0),
#                 dtype = dtype,
#                 name = "length_scale"
#             )
#             NULL
#         },
#         call = function(self, x, ...) {
#             x
#         },
#         kernel = bt$property(
#             reticulate::py_func(
#                 function(self) {
#                     tfp$math$psd_kernels$ExponentiatedQuadratic(
#                         amplitude = self$`_amplitude`,
#                         length_scale = self$`_length_scale`
#                     )
#                 }
#             )
#         )
#     )
# )
RBFKernelFn <- reticulate::PyClass(
        "KernelFn",
        inherit = tensorflow::tf$keras$layers$Layer,
        list(
            `__init__` = function(self, ...) {
                super()$`__init__`(...)
                kwargs <- list(...)
                dtype <- kwargs[["dtype"]]
                self$`_amplitude` = self$add_variable(
                    initializer = initializer_zeros(),
                    dtype = dtype,
                    trainable = TRUE,
                    name = "amplitude"
                )
                self$`_length_scale` = self$add_variable(
                    initializer = initializer_zeros(),
                    dtype = dtype,
                    trainable = TRUE,
                    name = "length_scale"
                )
                NULL
            },
            call = function(self, x, ...) {
                x
            },
            kernel = bt$property( # Use builtins module to access property
                reticulate::py_func(
                    function(self) {
                        tfp$math$psd_kernels$ExponentiatedQuadratic(
                            amplitude = tf$nn$softplus(array(0.1) * self$`_amplitude`),
                            length_scale = tf$nn$softplus(array(2) * self$`_length_scale`)
                        )
                    }
                )
            )
        )
    )
# RBFKernelFn()

# tfp$math$psd_kernels$ExponentiatedQuadratic(
#     # amplitude = tf$nn$softplus(array(0.1) * self$`_amplitude`),
#     amplitude = tf$Variable(initial_value = 1.0, dtype = tf$float64, name = "amplitude"),
#     #length_scale = tf$nn$softplus(array(1.0) * self$`_length_scale`)
#     length_scale = tf$Variable(initial_value = 1.0, dtype = tf$float64, name = "length_scale")
# )


MaternKernelFn <- reticulate::PyClass(
    "KernelFn",
    inherit = tensorflow::tf$keras$layers$Layer,
    list(
        `__init__` = function(self, dtype = NULL, trainable = TRUE, ...) {
            super()$`__init__`(trainable = trainable, ...)
            self$`_amplitude` <- self$add_variable(
                initializer = initializer_constant(value = 0.1),
                dtype = dtype,
                name = "amplitude"
            )
            self$`_length_scale` <- self$add_variable(
                initializer = initializer_constant(value = 1.0),
                dtype = dtype,
                name = "length_scale"
            )
            NULL
        },
        call = function(self, x, ...) {
            x
        },
        kernel = bt$property(
            reticulate::py_func(
                function(self) {
                    tfp$math$psd_kernels$MaternOneHalf(
                        amplitude = tf$nn$softplus(array(1) * self$`_amplitude`),
                        length_scale = tf$nn$softplus(array(1) * self$`_length_scale`)
                    )
                }
            )
        )
    )
)

MaternFiveHalvesKernelFn <- reticulate::PyClass(
    "KernelFn",
    inherit = tensorflow::tf$keras$layers$Layer,
    list(
        `__init__` = function(self, dtype = NULL, trainable = TRUE, ...) {
            super()$`__init__`(trainable = trainable, ...)
            self$`_amplitude` <- self$add_variable(
                initializer = initializer_zeros(),
                dtype = dtype,
                name = "amplitude"
            )
            self$`_length_scale` <- self$add_variable(
                initializer = initializer_zeros(),
                dtype = dtype,
                name = "length_scale"
            )
            NULL
        },
        call = function(self, x, ...) {
            x
        },
        kernel = bt$property(
            reticulate::py_func(
                function(self) {
                    tfp$math$psd_kernels$MaternFiveHalves(
                        amplitude = tf$nn$softplus(array(1) * self$`_amplitude`),
                        length_scale = tf$nn$softplus(array(2) * self$`_length_scale`)
                    )
                }
            )
        )
    )
)
