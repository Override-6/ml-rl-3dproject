use crate::player::Player;
use crate::ui::TriggerZoneText;
use bevy::prelude::{
    BevyError, Component, Entity, EventReader, Query, ResMut, Resource, Visibility, With,
};
use bevy_rapier3d::prelude::CollisionEvent;

#[derive(Component)]
pub struct TriggerZone;

#[derive(Resource)]
pub struct InTriggerZone(pub bool);

// Trigger zone detection system
pub fn check_trigger_zone(
    mut collision_events: EventReader<CollisionEvent>,
    player_query: Query<Entity, With<Player>>,
    trigger_query: Query<Entity, With<TriggerZone>>,
    mut text_query: Query<&mut Visibility, With<TriggerZoneText>>,
    mut in_zone: ResMut<InTriggerZone>,
) -> Result<(), BevyError> {
    let trigger = trigger_query.single()?;
    let players = player_query.iter().collect::<Vec<_>>();
    let player_one = players[0]; // controlled player
    let mut text = text_query.single_mut().ok();

    for event in collision_events.read() {
        match event {
            CollisionEvent::Started(e1, e2, _) => {
                if !(*e1 == trigger || *e2 == trigger) {
                    continue
                }
                let player = if *e1 == trigger { *e2 } else { *e1 };
                println!("Player {player} inside trigger zone");

                if player == player_one {
                    in_zone.0 = true;
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
                println!("Player {player} inside trigger zone");

                if player == player_one {
                    in_zone.0 = false;
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

