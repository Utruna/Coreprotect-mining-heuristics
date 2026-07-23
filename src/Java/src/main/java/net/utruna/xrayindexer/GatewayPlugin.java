package net.utruna.xrayindexer;

import net.utruna.xrayindexer.db.ReadOnlyCoreProtectDb;
import net.utruna.xrayindexer.http.GatewayServer;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.plugin.java.JavaPlugin;

import java.nio.file.Path;

/**
 * Passerelle HTTP en lecture seule au-dessus de la base CoreProtect.
 *
 * Le plugin ne fait qu'ouvrir une porte : il sert des tranches brutes de
 * co_block (paginées par rowid) et les tables de correspondance, en CSV gzippé.
 * Toute l'analyse x-ray vit ailleurs (pipeline Python), qui maintient un miroir
 * local alimenté par delta de rowid. Rien n'est jamais écrit dans CoreProtect.
 */
public final class GatewayPlugin extends JavaPlugin {

    private ReadOnlyCoreProtectDb coreProtectDb;
    private GatewayServer server;
    private String blockTable;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        FileConfiguration cfg = getConfig();

        String token = cfg.getString("gateway.token", "");
        if (token == null || token.isBlank() || "CHANGE_ME".equals(token)) {
            getLogger().severe("gateway.token n'est pas défini (ou laissé à CHANGE_ME).");
            getLogger().severe("La passerelle refuse de démarrer sans jeton : elle exposerait la base sans authentification.");
            getServer().getPluginManager().disablePlugin(this);
            return;
        }

        Path serverRoot = getServer().getWorldContainer().toPath().toAbsolutePath();
        Path coreProtectPath = serverRoot.resolve(
                cfg.getString("coreprotect-db-path", "plugins/CoreProtect/database.db"));

        this.blockTable = cfg.getString("table.block", "co_block");
        String userTable = cfg.getString("table.user", "co_user");
        String materialTable = cfg.getString("table.material-map", "co_material_map");
        String worldTable = cfg.getString("table.world", "co_world");

        String bind = cfg.getString("gateway.bind", "127.0.0.1");
        int port = cfg.getInt("gateway.port", 8787);
        int backlog = cfg.getInt("gateway.backlog", 0);
        int threads = cfg.getInt("gateway.threads", 4);
        int maxPageSize = cfg.getInt("gateway.max-page-size", 50000);

        try {
            this.coreProtectDb = new ReadOnlyCoreProtectDb(coreProtectPath);
            this.server = new GatewayServer(bind, port, backlog, token, threads,
                    coreProtectDb, blockTable, userTable, materialTable, worldTable,
                    maxPageSize, getLogger());
            this.server.start();
        } catch (Exception e) {
            getLogger().severe("Impossible de démarrer la passerelle : " + e.getMessage());
            getLogger().severe("Le plugin se désactive pour éviter tout comportement risqué.");
            getServer().getPluginManager().disablePlugin(this);
            return;
        }

        getLogger().info("Base CoreProtect ouverte en lecture seule stricte : " + coreProtectPath);
        getLogger().info("Passerelle en écoute sur " + bind + ":" + port
                + (bind.equals("127.0.0.1")
                    ? " (local uniquement — exposer via un tunnel/VPN pour un accès distant)."
                    : " (ATTENTION : bind non local — protège l'accès par firewall/VPN)."));
    }

    @Override
    public void onDisable() {
        if (server != null) {
            server.stop();
        }
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!command.getName().equalsIgnoreCase("xraygateway")) {
            return false;
        }
        if (coreProtectDb == null) {
            sender.sendMessage("§cPasserelle non initialisée.");
            return true;
        }
        try (var connection = coreProtectDb.open();
             var st = connection.createStatement();
             var rs = st.executeQuery("SELECT MAX(rowid) FROM " + blockTable + ";")) {
            long maxRowid = rs.next() ? rs.getLong(1) : 0L;
            sender.sendMessage("§7Passerelle XRay active. Tête co_block (rowid max) : §f" + maxRowid);
        } catch (Exception e) {
            sender.sendMessage("§cErreur : " + e.getMessage());
        }
        return true;
    }
}
