use crate::ai::input::InputSet;
use bevy::prelude::Resource;
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Resource, Clone, Encode, Decode)]
pub struct Script {
    pub(crate) inputs: Vec<InputSet>,
    pub(crate) debug_info: Vec<String>
}
