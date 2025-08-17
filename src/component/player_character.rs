use crate::player::{PLAYER_LASERS, Player, PlayerId};
use crate::sensor::ground_sensor::{GroundContact, GroundSensor};
use crate::sensor::objective::IsInObjective;
use crate::sensor::player_vibrissae::PlayerVibrissae;
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::color::palettes::basic::WHITE;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    Commands, Component, Entity, GlobalTransform, InheritedVisibility, Mesh, Mesh3d, Name, Query,
    ResMut, Text, TextColor, TextFont, Transform, With,
};
use bevy_math::prelude::Cuboid;
use bevy_rapier3d::dynamics::{
    AdditionalMassProperties, CoefficientCombineRule, LockedAxes, RigidBody, Velocity,
};
use bevy_rapier3d::geometry::{ActiveEvents, Collider, CollisionGroups, Friction, Group, Sensor};
use bevy_rapier3d::prelude::Sleeping;
use rand::Rng;

pub const PLAYER_WIDTH: f32 = 10.0;
pub const PLAYER_HEIGHT: f32 = 10.0;

#[derive(Component)]
pub struct PlayerInfoUI(pub(crate) Entity);

pub fn spawn_player_characters(
    players: Query<Entity, With<Player>>,
    mut commands: Commands,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let mut rng = rand::rng();
    for entity in players.iter() {
        let mut entity_commands = commands.entity(entity);
        entity_commands.insert((
            PlayerVibrissae::from(PLAYER_LASERS),
            IsInObjective(false),
            GroundContact(0),
            Transform::from_xyz(
                rng.random_range(-30.0..30.0),
                0.0,
                rng.random_range(-30.0..30.0),
            ),
            Velocity::default(),
            GlobalTransform::default(),
            InheritedVisibility::VISIBLE,
            RigidBody::Dynamic,
            Sleeping::disabled(),
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
            entity_commands.insert(Mesh3d(meshes.add(Cuboid::new(
                PLAYER_WIDTH,
                PLAYER_HEIGHT,
                PLAYER_WIDTH,
            ))));
        }
        if let Some(materials) = materials.as_mut() {
            entity_commands.insert(MeshMaterial3d(materials.add(Color::srgb(0.7, 1.0, 0.5))));
        }
        entity_commands.with_children(|parent| {
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

        if materials.is_some() {
            // Spawn Player Info UI if running in rendering mode
            commands.spawn((
                Text::new("<Info>"),
                PlayerInfoUI(entity),
                TextFont {
                    font_size: 20.0,
                    ..Default::default()
                },
                TextColor(WHITE.into()),
            ));
        }
    }
}
