"""
WebSocket message handlers.
"""

import logging
import asyncio
from typing import Dict

from ..core.simulator import FederatedSimulator
from ..config import SCENARIOS
from .manager import manager

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handle WebSocket messages and responses."""

    def __init__(self):
        """Initialize handler."""
        self.simulators: Dict[str, FederatedSimulator] = {}
        self.round_tasks: Dict[str, asyncio.Task] = {}

    async def handle_message(self, websocket, message: dict):
        """Handle incoming WebSocket message."""
        action = message.get("action")

        try:
            if action == "subscribe":
                await self._handle_subscribe(websocket, message)
            elif action == "start":
                await self._handle_start(websocket, message)
            elif action == "pause":
                await self._handle_pause(websocket, message)
            elif action == "resume":
                await self._handle_resume(websocket, message)
            elif action == "reset":
                await self._handle_reset(websocket, message)
            elif action == "add_bank":
                await self._handle_add_bank(websocket, message)
            elif action == "remove_bank":
                await self._handle_remove_bank(websocket, message)
            elif action == "inject_attack":
                await self._handle_inject_attack(websocket, message)
            elif action == "update_privacy":
                await self._handle_update_privacy(websocket, message)
            elif action == "get_status":
                await self._handle_get_status(websocket, message)
            else:
                await manager.send_personal_message(
                    {"type": "error", "message": f"Unknown action: {action}"},
                    websocket
                )

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await manager.send_personal_message(
                {"type": "error", "message": str(e)},
                websocket
            )

    async def _handle_subscribe(self, websocket, message):
        """Handle new client subscription."""
        scenario_name = message.get("scenario", "happy_path")

        # Create simulator for this scenario
        if scenario_name not in self.simulators:
            scenario = SCENARIOS.get(scenario_name, SCENARIOS["happy_path"])
            self.simulators[scenario_name] = FederatedSimulator(scenario)

        # Send current status
        simulator = self.simulators[scenario_name]
        status = simulator.get_status()

        await manager.send_personal_message({
            "type": "subscribed",
            "scenario": scenario_name,
            "status": status,
        }, websocket)

        logger.info(f"Client subscribed to scenario: {scenario_name}")

    async def _handle_start(self, websocket, message):
        """Handle start training."""
        scenario_name = message.get("scenario", "happy_path")
        speed = message.get("rounds_per_second", 0.5)

        if scenario_name not in self.simulators:
            await manager.send_personal_message(
                {"type": "error", "message": "Scenario not loaded"},
                websocket
            )
            return

        simulator = self.simulators[scenario_name]

        # Start training
        await simulator.start_training()

        # Start round loop
        if scenario_name not in self.round_tasks or self.round_tasks[scenario_name].done():
            self.round_tasks[scenario_name] = asyncio.create_task(
                self._round_loop(scenario_name, speed)
            )

        await manager.send_personal_message({
            "type": "started",
            "scenario": scenario_name,
        }, websocket)

        logger.info(f"Started training for scenario: {scenario_name}")

    async def _handle_pause(self, websocket, message):
        """Handle pause training."""
        scenario_name = message.get("scenario", "happy_path")

        if scenario_name in self.simulators:
            await self.simulators[scenario_name].pause_training()
            await manager.send_personal_message({
                "type": "paused",
                "scenario": scenario_name,
            }, websocket)

    async def _handle_resume(self, websocket, message):
        """Handle resume training."""
        scenario_name = message.get("scenario", "happy_path")

        if scenario_name in self.simulators:
            await self.simulators[scenario_name].resume_training()
            await manager.send_personal_message({
                "type": "resumed",
                "scenario": scenario_name,
            }, websocket)

    async def _handle_reset(self, websocket, message):
        """Handle reset training."""
        scenario_name = message.get("scenario", "happy_path")

        if scenario_name in self.simulators:
            # Cancel existing task
            if scenario_name in self.round_tasks:
                self.round_tasks[scenario_name].cancel()
                del self.round_tasks[scenario_name]

            await self.simulators[scenario_name].reset_training()

            # Broadcast reset
            await manager.broadcast({
                "type": "reset",
                "scenario": scenario_name,
                "status": self.simulators[scenario_name].get_status(),
            })

    async def _handle_add_bank(self, websocket, message):
        """Handle adding a new bank."""
        scenario_name = message.get("scenario", "happy_path")
        bank_config = message.get("bank")

        if scenario_name in self.simulators and bank_config:
            from ..config import BankConfig
            config = BankConfig(**bank_config)
            self.simulators[scenario_name].add_bank(config)

            await manager.broadcast({
                "type": "bank_added",
                "scenario": scenario_name,
                "bank": config.dict(),
            })

    async def _handle_remove_bank(self, websocket, message):
        """Handle removing a bank."""
        scenario_name = message.get("scenario", "happy_path")
        bank_id = message.get("bank_id")

        if scenario_name in self.simulators:
            self.simulators[scenario_name].remove_bank(bank_id)

            await manager.broadcast({
                "type": "bank_removed",
                "scenario": scenario_name,
                "bank_id": bank_id,
            })

    async def _handle_inject_attack(self, websocket, message):
        """Handle injecting an attack."""
        scenario_name = message.get("scenario", "happy_path")
        bank_id = message.get("bank_id")
        attack_type = message.get("attack_type", "sign_flip")

        if scenario_name in self.simulators:
            self.simulators[scenario_name].inject_attack(bank_id, attack_type)

            await manager.broadcast({
                "type": "attack_injected",
                "scenario": scenario_name,
                "bank_id": bank_id,
                "attack_type": attack_type,
            })

    async def _handle_update_privacy(self, websocket, message):
        """Handle updating privacy settings."""
        scenario_name = message.get("scenario", "happy_path")
        level = message.get("privacy_level")
        epsilon = message.get("epsilon")

        if scenario_name in self.simulators:
            if level is not None:
                self.simulators[scenario_name].update_privacy_level(level)

            await manager.broadcast({
                "type": "privacy_updated",
                "scenario": scenario_name,
                "privacy_level": level,
            })

    async def _handle_get_status(self, websocket, message):
        """Handle get status request."""
        scenario_name = message.get("scenario", "happy_path")

        if scenario_name in self.simulators:
            status = self.simulators[scenario_name].get_status()
            await manager.send_personal_message({
                "type": "status",
                "scenario": scenario_name,
                "status": status,
            }, websocket)

    async def _round_loop(self, scenario_name: str, speed: float):
        """Run training rounds in a loop."""
        simulator = self.simulators.get(scenario_name)
        if not simulator:
            return

        try:
            while simulator.is_running and not simulator.is_paused:
                # Run one round
                result = await simulator.run_round()

                if result:
                    # Broadcast to all clients
                    await manager.broadcast(result)

                    # Check if complete
                    if result.get("is_complete"):
                        await manager.broadcast({
                            "type": "training_complete",
                            "scenario": scenario_name,
                            "final_accuracy": result["global_accuracy"],
                        })
                        break

                # Wait for next round
                delay = 1.0 / speed if speed > 0 else 1.0
                await asyncio.sleep(delay)

        except asyncio.CancelledError:
            logger.info(f"Round loop cancelled for scenario: {scenario_name}")
        except Exception as e:
            logger.error(f"Error in round loop: {e}")


# Global handler instance
handler = WebSocketHandler()
