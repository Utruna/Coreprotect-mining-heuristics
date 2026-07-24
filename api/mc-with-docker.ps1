param(
    [ValidateSet("start", "stop", "restart")]
    [string]$Action = "start",

    [string]$ComposeFile = "",

    [string]$ApiService = "xrayindexer-api",

    [string]$ServerStartScript = "",

    [switch]$Build,

    [switch]$NoWait
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ComposeFile)) {
    $ComposeFile = Join-Path $PSScriptRoot "docker-compose.yml"
}

function Invoke-Compose {
    param([string[]]$Args)

    & docker compose -f $ComposeFile @Args
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose a échoué (code $LASTEXITCODE): $($Args -join ' ')"
    }
}

function Start-Api {
    $args = @("up", "-d")
    if ($Build) {
        $args += "--build"
    }
    $args += $ApiService
    Invoke-Compose -Args $args
}

function Stop-Api {
    Invoke-Compose -Args @("stop", $ApiService)
}

function Restart-Api {
    $args = @("up", "-d")
    if ($Build) {
        $args += "--build"
    }
    $args += $ApiService
    Invoke-Compose -Args $args
}

function Start-MinecraftServer {
    if ([string]::IsNullOrWhiteSpace($ServerStartScript)) {
        Write-Host "API démarrée. Aucune commande serveur fournie, rien d'autre à lancer."
        return
    }
    if (-not (Test-Path -LiteralPath $ServerStartScript)) {
        throw "Script de démarrage introuvable: $ServerStartScript"
    }

    Write-Host "Lancement du serveur Minecraft via: $ServerStartScript"
    if ($ServerStartScript.ToLowerInvariant().EndsWith(".ps1")) {
        & $ServerStartScript
    }
    else {
        & cmd.exe /c $ServerStartScript
    }
}

switch ($Action) {
    "start" {
        Start-Api
        if ($NoWait) {
            Write-Host "API démarrée (mode détaché)."
            break
        }

        try {
            Start-MinecraftServer
        }
        finally {
            if (-not [string]::IsNullOrWhiteSpace($ServerStartScript)) {
                Write-Host "Le processus serveur s'est arrêté. Extinction de l'API Docker..."
                try {
                    Stop-Api
                }
                catch {
                    Write-Warning "Arrêt API échoué: $($_.Exception.Message)"
                }
            }
        }
        break
    }
    "stop" {
        Stop-Api
        break
    }
    "restart" {
        Restart-Api
        break
    }
}
