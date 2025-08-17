use crate::player::Player;
use crate::ui::SuccessUIText;
use bevy::prelude::{
    BevyError, Component, Entity, EventReader, Query, Visibility, With,
};
use bevy_rapier3d::prelude::CollisionEvent;

#[derive(Component)]
pub struct Objective;

#[derive(Component)]
pub struct IsInObjective(pub bool);

// Trigger zone detection system
pub fn check_trigger_zone(
    mut collision_events: EventReader<CollisionEvent>,
    mut player_query: Query<Entity, With<Player>>,
    mut itz: Query<&mut IsInObjective>,
    trigger_query: Query<Entity, With<Objective>>,
    mut text_query: Query<&mut Visibility, With<SuccessUIText>>,
) -> Result<(), BevyError> {
    let trigger = trigger_query.single()?;
    let players = player_query.iter_mut().next();
    let player_one = players.unwrap(); // controlled player
    let mut text = text_query.single_mut().ok();

    for event in collision_events.read() {
        match event {
            CollisionEvent::Started(e1, e2, _) => {
                if !(*e1 == trigger || *e2 == trigger) {
                    continue
                }
                let player = if *e1 == trigger { *e2 } else { *e1 };

                if let Ok(mut in_zone) = itz.get_mut(player) {
                    in_zone.0 = true;
                }

                if player == player_one {
                    if let Some(text) = text.as_mut() {
                        *text.as_mut() = Visibility::Visible;
                    }
                }
            }
            CollisionEvent::Stopped(e1, e2, _) => {
                if !(*e1 == trigger || *e2 == trigger) {
                    continue
                }
                let player = if *e1 == trigger { *e2 } else { *e1 };

                if let Ok(mut in_zone) = itz.get_mut(player) {
                    in_zone.0 = false;
                }

                if player == player_one {
                    if let Some(text) = text.as_mut() {
                        *text.as_mut() = Visibility::Hidden;
                    }
                }
                println!("Player {player} outside trigger zone")
            }
        }
    }
    Ok(())
}

