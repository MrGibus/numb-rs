//! Macros for use in other modules which shouldn't be exported.

/// Implements a trait with some common code for a list of types
macro_rules! impl_multiple {
    () => ();

    // standard implementation with no internal code
    ($trait:ident for $($type:ty)*) => {
        $(
            impl $trait for $type {}
        )*
    };

    // standard implementation with common internal code for a list of types
    ($trait:ident => $body:tt for $($type:ty)*) => {
        $(
            impl $trait for $type $body
        )*
    };
}

