use crate::ai::input::{Input, InputSet};
use crate::ai::script::Script;
use bevy::prelude::Resource;
use bincode::Encode;

#[derive(Debug, Resource, Encode)]
pub struct InputRecorder<const N: usize> {
    buff: [InputSet; N],
    debug_buff: [String; N],
    len: usize,
}



impl<const N: usize> InputRecorder<N> {
    pub fn new() -> Self {
        Self {
            buff: [Input::Noop as InputSet; N],
            debug_buff: [const { String::new() }; N],
            len: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.len >= N
    }

    pub fn record(&mut self, input: InputSet, debug: String) {
        if self.is_full() {
            panic!("Input buffer overflow");
        }
        self.buff[self.len] = input;
        self.debug_buff[self.len] = debug;
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<const N: usize> Into<Script> for InputRecorder<N> {
    fn into(self) -> Script {
        (&self).into()
    }
}

impl<const N: usize> Into<Script> for &InputRecorder<N> {
    fn into(self) -> Script {
        Script {
            inputs: Vec::from(&self.buff[..self.len]),
            debug_info: Vec::from(&self.debug_buff[..self.len]),
        }
    }
}