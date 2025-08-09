use bevy::prelude::{BevyError, ChildOf, Component, Entity, EventReader, Query, With};
use bevy_rapier3d::prelude::*;

#[derive(Component, Default, Debug)]
pub struct GroundContact(pub u32);

#[derive(Component)]
pub struct GroundSensor;

pub fn ground_sensor_events(
    mut contact_events: EventReader<CollisionEvent>,
    sensor_query: Query<(Entity, &ChildOf), With<GroundSensor>>,
    mut players_contacts: Query<&mut GroundContact>,
) -> Result<(), BevyError> {
    for event in contact_events.read() {
        match event {
            CollisionEvent::Started(e1, e2, _) | CollisionEvent::Stopped(e1, e2, _) => {
                for (sensor_entity, parent) in sensor_query.iter() {
                    if *e1 != sensor_entity && *e2 != sensor_entity {
                        continue;
                    }

                    if let Ok(mut ground_contact) = players_contacts.get_mut(parent.parent()) {
                        match event {
                            CollisionEvent::Started(_, _, _) => {
                                ground_contact.0 += 1;
                            }
                            CollisionEvent::Stopped(_, _, _) => {
                                ground_contact.0 = ground_contact.0.saturating_sub(1);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
