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
    drone_query: Query<Entity, With<Player>>,
    trigger_query: Query<Entity, With<TriggerZone>>,
    mut text_query: Query<&mut Visibility, With<TriggerZoneText>>,
    mut in_zone: ResMut<InTriggerZone>,
) -> Result<(), BevyError> {
    let drone = drone_query.single()?;
    let trigger = trigger_query.single()?;
    let mut text = text_query.single_mut()?;

    for event in collision_events.read() {
        match event {
            CollisionEvent::Started(e1, e2, _) => {
                if (*e1 == drone && *e2 == trigger) || (*e1 == trigger && *e2 == drone) {
                    in_zone.0 = true;
                    *text = Visibility::Visible;
                }
            }
            CollisionEvent::Stopped(e1, e2, _) => {
                if (*e1 == drone && *e2 == trigger) || (*e1 == trigger && *e2 == drone) {
                    in_zone.0 = false;
                    *text = Visibility::Hidden;
                }
            }
        }
    }
    Ok(())
}
