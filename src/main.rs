mod camera_controller;
mod map;
mod objective;
mod player;
mod ui;
// unused
mod component;
mod drone;
mod sensor;

use bevy::ecs::schedule::ScheduleLabel;
use crate::camera_controller::{camera_follow, mouse_look};
use crate::component::arrow::{spawn_arrow_assets, spawn_arrows_to_players};
use crate::map::setup_map;
use crate::objective::{check_trigger_zone, InTriggerZone};
use crate::player::{move_player, player_raycast_update, setup_player};
use crate::sensor::ground_sensor::ground_sensor_events;
use crate::ui::{setup_ui, update_stats_text};
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::dynamics::IntegrationParameters;


#[derive(ScheduleLabel, Clone, Hash, PartialEq, Eq, Debug)]
pub struct PostPhysics;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    watch_for_changes_override: Some(true),
                    ..default()
                })
                .add(RapierPhysicsPlugin::<NoUserData>::default())
                .add(RapierDebugRenderPlugin::default())
                .add(MeshPickingPlugin),
        )
        .add_systems(
            Startup,
            (setup, setup_ui, setup_map, setup_player, spawn_arrow_assets),
        )
        .add_systems(PostStartup, spawn_arrows_to_players)
        .add_systems(
            Update,
            (
                move_player.before(PhysicsSet::SyncBackend),
                mouse_look,
                update_stats_text,
                check_trigger_zone,
                ground_sensor_events,
                camera_follow,
            ),
        )
        .add_systems(Update, (player_raycast_update.after(PhysicsSet::Writeback)))
        .insert_resource(
            RapierContextInitialization::InitializeDefaultRapierContext {
                integration_parameters: IntegrationParameters {
                    length_unit: 1.0, // standard scaling
                    ..default()
                },
                rapier_configuration: RapierConfiguration {
                    // gravity: Vect::new(0.0, -2000.0, 0.0),
                    ..RapierConfiguration::new(100f32)
                },
            },
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.insert_resource(InTriggerZone(false));
}
