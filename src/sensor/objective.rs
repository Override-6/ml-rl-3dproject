use bevy::prelude::{
    BevyError, Component, Entity, EventReader, Query, With,
};
use bevy_rapier3d::prelude::CollisionEvent;

#[derive(Component)]
pub struct Objective;

#[derive(Component)]
pub struct IsInObjective(pub bool);

// Trigger zone detection system
pub fn check_trigger_zone(
    mut collision_events: EventReader<CollisionEvent>,
    mut itz: Query<&mut IsInObjective>,
    trigger_query: Query<Entity, With<Objective>>,
) -> Result<(), BevyError> {
    let trigger = trigger_query.single()?;
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
            }
            CollisionEvent::Stopped(e1, e2, _) => {
                if !(*e1 == trigger || *e2 == trigger) {
                    continue
                }
                let player = if *e1 == trigger { *e2 } else { *e1 };

                if let Ok(mut in_zone) = itz.get_mut(player) {
                    in_zone.0 = false;
                }
            }
        }
    }
    Ok(())
}

