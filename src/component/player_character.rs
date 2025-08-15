use crate::player::{Player, PLAYER_LASERS};
use crate::sensor::ground_sensor::{GroundContact, GroundSensor};
use crate::sensor::player_vibrissae::PlayerVibrissae;
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    Commands, Entity, GlobalTransform, InheritedVisibility, Mesh, Mesh3d, Name, Query, ResMut,
    Transform, With,
};
use bevy_math::prelude::Cuboid;
use bevy_math::Vec3;
use bevy_rapier3d::dynamics::{
    AdditionalMassProperties, CoefficientCombineRule, LockedAxes, RigidBody, Velocity,
};
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::geometry::{ActiveEvents, Collider, CollisionGroups, Friction, Group, Sensor};
use rand::Rng;
use crate::sensor::objective::IsInObjective;

pub const PLAYER_WIDTH: f32 = 10.0;
pub const PLAYER_HEIGHT: f32 = 10.0;

pub fn spawn_player_character(
    players: Query<Entity, With<Player>>,
    mut commands: Commands,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let mut rng = rand::rng();
    for player in players.iter() {
        let mut player = commands.entity(player);
        player.insert((
            PlayerVibrissae::from(PLAYER_LASERS),
            IsInObjective(false),
            GroundContact(0),
            Transform::from_xyz(rng.random_range(-30.0..30.0), 0.0, rng.random_range(-30.0..30.0)),
            Velocity::default(),
            GlobalTransform::default(),
            InheritedVisibility::VISIBLE,
            RigidBody::Dynamic,
            AdditionalMassProperties::Mass(200.0),
            Collider::cuboid(PLAYER_WIDTH / 2.0, PLAYER_HEIGHT / 2.0, PLAYER_WIDTH / 2.0),
            CollisionGroups::new(Group::GROUP_1, Group::GROUP_2),
            LockedAxes::ROTATION_LOCKED ^ LockedAxes::ROTATION_LOCKED_Y,
            Friction {
                coefficient: 0.0,
                combine_rule: CoefficientCombineRule::Min,
            },
        ));
        if let Some(meshes) = meshes.as_mut() {
            player.insert(Mesh3d(meshes.add(Cuboid::new(
                PLAYER_WIDTH,
                PLAYER_HEIGHT,
                PLAYER_WIDTH,
            ))));
        }
        if let Some(materials) = materials.as_mut() {
            player.insert(MeshMaterial3d(materials.add(Color::srgb(0.7, 1.0, 0.5))));
        }
        player.with_children(|parent| {
            // spawn ground detection
            parent.spawn((
                Name::new("GroundSensor"),
                GroundSensor,
                Collider::cuboid(PLAYER_WIDTH / 2.0, 0.1, PLAYER_WIDTH / 2.0),
                Sensor,
                CollisionGroups::new(Group::GROUP_1, Group::GROUP_2),
                ActiveEvents::COLLISION_EVENTS,
                Transform::from_xyz(0.0, -(PLAYER_HEIGHT / 2.0), 0.0),
                GlobalTransform::default(),
            ));
        });
    }
}
