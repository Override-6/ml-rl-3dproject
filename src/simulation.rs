use crate::ai::player::{AIPlayer, AIPlayerId};
use crate::ai::script::Script;
use crate::human::player::HumanPlayer;
use crate::player::Player;
use bevy::prelude::{Commands, Res, Resource};
use std::ops::Deref;
use avian3d::prelude::PhysicsLayer;

pub const TICK_RATE: f32 = 5.0;
pub const DELTA_TIME: f32 = 1.0 / TICK_RATE;

#[derive(PhysicsLayer, Default)]
pub enum GameLayer {
    #[default]
    World,
    Player,
    Laser,
}

#[derive(Resource)]
pub enum Simulation {
    Simulation {
        num_ai_players: usize,
        script: Script,
    },
    Game,
}

pub fn spawn_players(mut commands: Commands, simulation: Res<Simulation>) {
    match simulation.deref() {
        Simulation::Simulation { num_ai_players, .. } => {
            commands.spawn_batch(
                (0..*num_ai_players)
                    .into_iter()
                    .map(|id| (Player, AIPlayer::new(AIPlayerId(id)))),
            );
        }
        Simulation::Game => {
            commands.spawn((Player, HumanPlayer));
        }
    }
}
