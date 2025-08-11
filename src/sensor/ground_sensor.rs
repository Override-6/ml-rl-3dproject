use crate::component::player_character::PLAYER_HEIGHT;
use crate::player::Player;
use crate::sensor::player_vibrissae::PlayerVibrissae;
use bevy::prelude::{BevyError, Component, Query, With};

#[derive(Component, Default, Debug)]
pub struct GroundContact(pub bool);

pub fn ground_sensor_events(
    mut players_contacts: Query<(&mut GroundContact, &PlayerVibrissae), With<Player>>,
) -> Result<(), BevyError> {
    players_contacts
        .par_iter_mut()
        .for_each(|(mut contact, vibrissae)| {
            let ground_laser = vibrissae.ground_sensor();
            let Some(hit) = &ground_laser.hit else { return };
            contact.0 = hit.distance <= (PLAYER_HEIGHT / 2.0);
        });
    Ok(())
}
