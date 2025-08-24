use std::ops::{BitAnd, BitOr, BitOrAssign};

pub struct Input {
    pub(crate) pulse: PulseInputSet,
    pub(crate) laser_dir: f32
}

pub type PulseInputSet = u8;

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PulseInput {
    #[default]
    Noop = 0u8,
    Forward = 1u8,
    Backward = 2u8,
    Left = 4u8,
    Right = 8u8,
    Jump = 16u8,
    TurnRight = 32u8,
    TurnLeft = 64u8,
    Interact = 128u8,
}

impl BitOrAssign<PulseInput> for u8 {
    fn bitor_assign(&mut self, rhs: PulseInput) {
        *self |= rhs as u8;
    }
}

impl BitOr for PulseInput {
    type Output = PulseInputSet;

    fn bitor(self, rhs: Self) -> Self::Output {
        self as u8 | rhs as u8
    }
}

impl BitAnd<PulseInput> for PulseInputSet {
    type Output = PulseInputSet;

    fn bitand(self, rhs: PulseInput) -> Self::Output {
        self as u8 & rhs as u8
    }
}
