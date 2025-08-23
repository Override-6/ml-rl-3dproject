use crate::player::{get_player_position, Player, PLAYER_LASERS};
use crate::sensor::ground_sensor::{GroundContact, GroundSensor};
use crate::sensor::objective::IsInObjective;
use crate::sensor::player_vibrissae::PlayerVibrissae;
use bevy::asset::Assets;
use bevy::color::palettes::basic::WHITE;
use bevy::color::Color;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    Commands, Component, Entity, GlobalTransform, InheritedVisibility, Mesh, Mesh3d, Name, Query,
    ResMut, Text, TextColor, TextFont, Transform, With,
};
use bevy_math::prelude::Cuboid;
use bevy_rapier3d::dynamics::{
    AdditionalMassProperties, CoefficientCombineRule, LockedAxes, RapierRigidBodyHandle,
    RigidBody, Velocity,
};
use bevy_rapier3d::geometry::{ActiveEvents, Collider, CollisionGroups, Friction, Group, Sensor};
use bevy_rapier3d::plugin::WriteRapierContext;
use bevy_rapier3d::prelude::Sleeping;
use bevy_rapier3d::rapier::prelude as rapier;

pub const PLAYER_WIDTH: f32 = 10.0;
pub const PLAYER_HEIGHT: f32 = 10.0;

#[derive(Component)]
pub struct PlayerInfoUI(pub(crate) Entity);

pub fn spawn_player_characters(
    players: Query<Entity, With<Player>>,
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let mut rng = rand::rng();
    let player_material = materials.map(|mut m| MeshMaterial3d(m.add(Color::srgb(0.3, 0.0, 0.25))));
    let player_mesh =
        meshes.map(|mut m| Mesh3d(m.add(Cuboid::new(PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH))));
    for entity in players.iter() {
        let mut entity_commands = commands.entity(entity);
        entity_commands.insert((
            PlayerVibrissae::from(PLAYER_LASERS),
            IsInObjective(false),
            GroundContact(0),
            Transform::from_translation(get_player_position()),
            Velocity::default(),
            GlobalTransform::default(),
            InheritedVisibility::VISIBLE,
            RigidBody::Dynamic,
            Sleeping::disabled(),
            AdditionalMassProperties::Mass(200.0),
            Collider::cuboid(PLAYER_WIDTH / 2.0, PLAYER_HEIGHT / 2.0, PLAYER_WIDTH / 2.0),
            CollisionGroups::new(Group::GROUP_1, Group::GROUP_2 | Group::GROUP_3),
            LockedAxes::ROTATION_LOCKED ^ LockedAxes::ROTATION_LOCKED_Y,
            Friction {
                coefficient: 0.0,
                combine_rule: CoefficientCombineRule::Min,
            },
        ));
        if let Some(mesh) = player_mesh.as_ref() {
            entity_commands.insert(mesh.clone());
        }
        if let Some(material) = player_material.as_ref() {
            entity_commands.insert(material.clone());
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

        if player_material.is_some() {
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

pub fn freeze_frozen_players(
    player_q: Query<(&RapierRigidBodyHandle, &Player, &RigidBody)>,
    mut rapier_ctx: WriteRapierContext,
) -> bevy::prelude::Result<()> {
    let mut context = rapier_ctx.single_mut()?;
    for (handle, player, rb) in player_q.iter() {
        if !player.freeze || *rb == RigidBody::Fixed {
            continue;
        }

        let handle = handle.0;

        if let Some(rb) = context.rigidbody_set.bodies.get_mut(handle) {
            rb.set_body_type(rapier::RigidBodyType::Fixed, false);
        }
    }

    Ok(())
}
