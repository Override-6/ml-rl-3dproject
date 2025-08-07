use std::ops::{BitAnd, BitOr, BitOrAssign};


pub type InputSet = u8;


#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Input {
    #[default]
    Noop = 0u8,
    Forward = 1u8,
    Backward = 2u8,
    Left = 4u8,
    Right = 8u8,
    Jump = 16u8,
    TurnRight = 32u8,
    TurnLeft = 64u8,
}

impl BitOrAssign<Input> for u8 {
    fn bitor_assign(&mut self, rhs: Input) {
        *self |= rhs as u8;
    }
}

impl BitOr for Input {
    type Output = InputSet;

    fn bitor(self, rhs: Self) -> Self::Output {
        self as u8 | rhs as u8
    }
}

impl BitAnd<Input> for InputSet {
    type Output = InputSet;

    fn bitand(self, rhs: Input) -> Self::Output {
        self as u8 & rhs as u8
    }
}