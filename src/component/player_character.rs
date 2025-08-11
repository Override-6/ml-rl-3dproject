use crate::player::Player;
use crate::sensor::ground_sensor::{GroundContact};
use crate::sensor::player_vibrissae::PlayerVibrissae;
use avian3d::prelude::{CoefficientCombine, Collider, CollisionLayers, DebugRender, Friction, GravityScale, LayerMask, LockedAxes, Mass, RigidBody, Sensor};
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    Commands, Entity, GlobalTransform, InheritedVisibility, Mesh, Mesh3d, Name, Query, ResMut,
    Transform, With,
};
use bevy_math::prelude::Cuboid;
use bevy_math::Dir3;
use rand::Rng;
use crate::simulation::GameLayer;

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
            PlayerVibrissae::from_vec(vec![
                Dir3::Z,
                Dir3::NEG_Z,
                Dir3::X,
                Dir3::NEG_X,
            ]),
            Transform::from_xyz(rng.random_range(-30.0..30.0), 0.0, rng.random_range(-30.0..30.0)),
            // Transform::from_xyz(0.0, PLAYER_HEIGHT / 2.0, 0.0),
            GlobalTransform::default(),
            InheritedVisibility::VISIBLE,
            GroundContact(false),
            RigidBody::Dynamic,
            GravityScale(100.0),
            Collider::cuboid(PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH),
            // CollisionGroups::new(Group::GROUP_1, Group::ALL ^ Group::GROUP_1),
            LockedAxes::ROTATION_LOCKED.unlock_rotation_y(),
            Friction::ZERO.with_combine_rule(CoefficientCombine::Min),
            CollisionLayers::new(GameLayer::Player, LayerMask::ALL ^ GameLayer::Player),
            DebugRender::none(),
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
    }
}
