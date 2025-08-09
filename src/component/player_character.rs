use crate::player::Player;
use crate::sensor::ground_sensor::GroundSensor;
use crate::sensor::player_vibrissae::PlayerVibrissae;
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    Commands, Component, Entity, GlobalTransform, Mesh, Mesh3d
    , Name, Query, ResMut, Transform, With,
};
use bevy_math::prelude::Cuboid;
use bevy_math::Vec3;
use bevy_rapier3d::dynamics::LockedAxes;
use bevy_rapier3d::geometry::{ActiveEvents, Collider, Sensor};
use bevy_rapier3d::na::DimAdd;

#[derive(Component)]
pub struct PlayerCharacter;

pub const PLAYER_WIDTH: f32 = 10.0;
pub const PLAYER_HEIGHT: f32 = 10.0;

pub fn spawn_player_character(
    players: Query<Entity, With<Player>>,
    mut commands: Commands,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    for player in players.iter() {
        commands.entity(player).with_children(|parent| {
            let mut entity = parent.spawn((
                Name::new("PlayerCharacter"),
                PlayerCharacter,
                PlayerVibrissae::from_vec(vec![Vec3::NEG_Z, Vec3::Z, Vec3::X, Vec3::NEG_X, Vec3::NEG_Y])
            ));

            if let Some(meshes) = meshes.as_mut() {
                entity.insert(Mesh3d(meshes.add(Cuboid::new(
                    PLAYER_WIDTH,
                    PLAYER_HEIGHT,
                    PLAYER_WIDTH,
                ))));
            }
            if let Some(materials) = materials.as_mut() {
                entity.insert(MeshMaterial3d(materials.add(Color::srgb(0.7, 1.0, 0.5))));
            }
        });
        commands.entity(player).with_children(|parent| {
            // spawn ground detection
            parent.spawn((
                Name::new("GroundSensor"),
                GroundSensor,
                Collider::cuboid(PLAYER_WIDTH / 2.0, 0.1, PLAYER_WIDTH / 2.0),
                Sensor,
                ActiveEvents::COLLISION_EVENTS,
                Transform::from_xyz(0.0, -(PLAYER_HEIGHT / 2.0), 0.0),
                GlobalTransform::default(),
            ));
        });
    }
}

