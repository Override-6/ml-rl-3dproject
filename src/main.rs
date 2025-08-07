mod map;
mod objective;
mod player;
mod ui;
// unused
mod ai;
mod component;
mod drone;
mod human;
mod sensor;

use crate::ai::input_recorder::InputRecorder;
use crate::ai::player::{follow_script, spawn_ai_player};
use crate::ai::script::Script;
use crate::component::arrow::{spawn_arrow_resource, spawn_arrows_to_players};
use crate::component::player_character::{player_raycast_update, spawn_player_character};
use crate::human::camera_controller::{camera_follow, mouse_look};
use crate::human::player::{move_player, spawn_human_player};
use crate::map::setup_map;
use crate::objective::{InTriggerZone, check_trigger_zone};
use crate::sensor::ground_sensor::ground_sensor_events;
use crate::ui::{setup_ui, update_stats_text};
use bevy::ecs::schedule::ScheduleLabel;
use bevy::pbr::DirectionalLightShadowMap;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::dynamics::IntegrationParameters;
use bincode::{encode_into_slice, encode_into_writer};
use std::cmp::PartialEq;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};

#[derive(ScheduleLabel, Clone, Hash, PartialEq, Eq, Debug)]
pub struct PostPhysics;

const TICK_RATE: f32 = 5.0;

fn main() {
    let file = File::open("script.bin");
    let mut script = None;
    if let Ok(mut file) = file {
        let mut script_data = Vec::new();
        file.read_to_end(&mut script_data).unwrap();
        let decoded_script: Script = bincode::decode_from_slice(&script_data, bincode::config::standard()).unwrap().0;
        script = Some(decoded_script);
    }

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(AssetPlugin {
                watch_for_changes_override: Some(true),
                ..default()
            })
            .add(RapierPhysicsPlugin::<NoUserData>::default())
            .add(RapierDebugRenderPlugin::default())
            .add(MeshPickingPlugin),
    )
    .add_systems(Startup, (setup, setup_ui, setup_map, spawn_arrow_resource))
    .add_systems(
        PostStartup,
        (
            spawn_player_character,
            spawn_arrows_to_players.after(spawn_player_character),
        ),
    )
    .add_systems(
        FixedUpdate,
        (
            player_raycast_update,
            update_stats_text,
            check_trigger_zone,
            ground_sensor_events,
        ),
    )
    .add_systems(Update, (mouse_look, camera_follow))
    .add_systems(Update, cleanup_on_exit)
    .insert_resource(
        RapierContextInitialization::InitializeDefaultRapierContext {
            integration_parameters: IntegrationParameters {
                dt: TICK_RATE,
                max_ccd_substeps: 1,
                ..default()
            },
            rapier_configuration: RapierConfiguration {
                ..RapierConfiguration::new(100f32)
            },
        },
    )
    .insert_resource(DirectionalLightShadowMap { size: 100 })
    .insert_resource(Time::<Fixed>::from_hz(TICK_RATE as f64))
    .insert_resource(InputRecorder::<100>::new());

    if let Some(script) = script {
        app.insert_resource(script);
        app.add_systems(Startup, (spawn_ai_player));
        app.add_systems(FixedUpdate, (follow_script));
    } else {
        app.add_systems(Startup, (spawn_human_player));
        app.add_systems(FixedUpdate, (move_player));
    }

    app.run();
}

fn setup(mut commands: Commands) {
    commands.insert_resource(InTriggerZone(false));
}

fn cleanup_on_exit(
    mut events: EventReader<AppExit>,
    recorded_input: Res<InputRecorder<100>>,
) -> Result<(), BevyError> {
    if let Some(event) = events.read().next() {
        info!("App is exiting with status: {:?}", event);

        let recorded_input: Script = recorded_input.as_ref().into();

        let mut file = File::create("script.bin")?;
        let mut buff = [0u8; 100000];
        encode_into_slice(&recorded_input, &mut buff, bincode::config::standard())?;
        file.write_all(&buff)?;
    }
    Ok(())
}
