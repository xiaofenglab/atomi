from importlib.resources import files
from string import Template


def render_submit_script(
    scheduler: str,
    profile_name: str,
    job_name: str,
    command: str,
) -> str:
    """Render a scheduler script from profile data and a scheduler template."""
    profile_path = files("atomi").joinpath(
        "templates", "profiles", f"{profile_name}.yaml"
    )
    template_path = files("atomi").joinpath(
        "templates", "schedulers", f"{scheduler}.sh"
    )

    profile = _read_simple_profile(profile_path.read_text(encoding="utf-8"))
    template = Template(template_path.read_text(encoding="utf-8"))
    values = {
        **profile,
        "job_name": job_name,
        "command": command,
    }
    return template.safe_substitute(values)


def _read_simple_profile(text: str) -> dict[str, str]:
    """Read the small profile format used by this package without external dependencies."""
    values: dict[str, str] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value == "|":
            block: list[str] = []
            while index < len(lines) and lines[index].startswith("  "):
                block.append(lines[index][2:])
                index += 1
            values[key] = "\n".join(block)
        else:
            values[key] = raw_value.strip("\"'")

    return values
