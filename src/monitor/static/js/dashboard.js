// Dashboard JavaScript

class MonitorDashboard {
    constructor() {
        this.socket = null;
        this.trainingChart = null;
        this.systemChart = null;
        this.isConnected = false;
        this.updateInterval = null;
        this.liveMode = true;
        this.currentExperimentId = null;
        this.experiments = [];
        this.loadingModal = null;
        
        this.init();
    }
    
    init() {
        this.setupSocket();
        this.setupCharts();
        this.setupEventListeners();
        this.loadExperiments();
        this.startDataUpdates();
    }
    
    setupSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.isConnected = true;
            if (this.liveMode) {
                this.updateStatusIndicator('Connected', 'success');
            }
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnected = false;
            if (this.liveMode) {
                this.updateStatusIndicator('Disconnected', 'danger');
            }
        });
        
        this.socket.on('training_update', (data) => {
            this.updateTrainingData(data);
        });
        
        this.socket.on('system_update', (data) => {
            this.updateSystemData(data);
        });
        
        this.socket.on('progress_update', (data) => {
            this.updateProgressData(data);
        });
        
        this.socket.on('alert_update', (data) => {
            this.addAlert(data);
        });
    }
    
    setupCharts() {
        // Training Metrics Chart
        const trainingCtx = document.getElementById('trainingChart').getContext('2d');
        this.trainingChart = new Chart(trainingCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Loss',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Learning Rate',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        tension: 0.1,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Throughput',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        tension: 0.1,
                        yAxisID: 'y2'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Loss'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Learning Rate'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    },
                    y2: {
                        type: 'linear',
                        display: false,
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
        
        // System Metrics Chart
        const systemCtx = document.getElementById('systemChart').getContext('2d');
        this.systemChart = new Chart(systemCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU Usage (%)',
                        data: [],
                        borderColor: 'rgb(255, 159, 64)',
                        backgroundColor: 'rgba(255, 159, 64, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Memory Usage (%)',
                        data: [],
                        borderColor: 'rgb(153, 102, 255)',
                        backgroundColor: 'rgba(153, 102, 255, 0.1)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Usage (%)'
                        },
                        min: 0,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
    }
    
    setupEventListeners() {
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.refreshData();
        });
        
        // Control buttons
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startMonitoring();
        });
        
        document.getElementById('stop-btn').addEventListener('click', () => {
            this.stopMonitoring();
        });
        
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetMonitoring();
        });
        
        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportData();
        });

        document.getElementById('experiment-select').addEventListener('change', (event) => {
            this.handleExperimentChange(event.target.value);
        });
    }
    
    startDataUpdates() {
        this.updateInterval = setInterval(() => {
            if (this.liveMode && this.isConnected) {
                this.socket.emit('request_update');
            }
        }, 2000); // Update every 2 seconds
    }
    
    async refreshData() {
        try {
            this.showLoading(true);

            if (!this.liveMode && this.currentExperimentId) {
                const snapshot = await this.fetchExperimentSnapshot(this.currentExperimentId);
                if (snapshot) {
                    this.applyExperimentSnapshot(snapshot);
                }
                await this.loadExperiments();
                return;
            }

            // Fetch all data
            const [status, training, system, progress, alerts, performance] = await Promise.all([
                this.fetchData('/api/status'),
                this.fetchData('/api/training_metrics'),
                this.fetchData('/api/system_metrics'),
                this.fetchData('/api/progress'),
                this.fetchData('/api/alerts'),
                this.fetchData('/api/performance_stats')
            ]);
            
            // Update UI
            this.updateStatus(status);
            this.updateTrainingChart(training);
            this.updateSystemChart(system);
            this.updateProgress(progress);
            this.updateAlerts(alerts);
            this.updatePerformanceStats(performance);
            this.loadExperiments();
            
        } catch (error) {
            console.error('Error refreshing data:', error);
            this.showAlert('Error refreshing data: ' + error.message, 'danger');
        } finally {
            this.showLoading(false);
        }
    }
    
    async fetchData(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }
    
    updateStatus(data) {
        document.getElementById('monitor-status').textContent = 
            data.monitor_active ? 'Active' : 'Inactive';
    }
    
    updateTrainingChart(data) {
        if (!data || data.length === 0) {
            this.trainingChart.data.labels = [];
            this.trainingChart.data.datasets.forEach(dataset => {
                dataset.data = [];
            });
            this.trainingChart.update('none');
            return;
        }
        
        const labels = data.map(d => new Date(d.timestamp).toLocaleTimeString());
        const lossData = data.map(d => d.loss);
        const lrData = data.map(d => d.learning_rate);
        const throughputData = data.map(d => d.throughput);
        
        this.trainingChart.data.labels = labels;
        this.trainingChart.data.datasets[0].data = lossData;
        this.trainingChart.data.datasets[1].data = lrData;
        this.trainingChart.data.datasets[2].data = throughputData;
        this.trainingChart.update('none');
    }
    
    updateSystemChart(data) {
        if (!data || data.length === 0) {
            this.systemChart.data.labels = [];
            this.systemChart.data.datasets.forEach(dataset => {
                dataset.data = [];
            });
            this.systemChart.update('none');
            return;
        }
        
        const labels = data.map(d => new Date(d.timestamp).toLocaleTimeString());
        const cpuData = data.map(d => d.cpu_percent);
        const memoryData = data.map(d => d.memory_percent);
        
        this.systemChart.data.labels = labels;
        this.systemChart.data.datasets[0].data = cpuData;
        this.systemChart.data.datasets[1].data = memoryData;
        this.systemChart.update('none');
    }
    
    updateProgress(data) {
        if (!data || Object.keys(data).length === 0) {
            const progressBar = document.getElementById('progress-bar');
            const progressText = document.getElementById('progress-text');
            progressBar.style.width = '0%';
            progressText.textContent = '0%';
            document.getElementById('progress-percent').textContent = '0%';
            document.getElementById('current-epoch').textContent = '0';
            document.getElementById('current-step').textContent = '0';
            document.getElementById('eta-time').textContent = '--:--';
            document.getElementById('throughput-value').textContent = '0';
            return;
        }
        
        const progressPercent = data.total_progress || 0;
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        
        progressBar.style.width = `${progressPercent}%`;
        progressText.textContent = `${progressPercent.toFixed(1)}%`;
        
        document.getElementById('progress-percent').textContent = `${progressPercent.toFixed(1)}%`;
        document.getElementById('current-epoch').textContent = data.current_epoch || 0;
        document.getElementById('current-step').textContent = data.current_step || 0;
        document.getElementById('eta-time').textContent = data.eta_total || '--:--';
        document.getElementById('throughput-value').textContent = 
            data.throughput ? data.throughput.toFixed(1) : '0';
    }
    
    updateAlerts(data) {
        if (!data || data.length === 0) {
            document.getElementById('alerts-list').innerHTML = '<p class="text-muted">No alerts</p>';
            document.getElementById('alert-count').textContent = '0';
            return;
        }
        
        const alertsList = document.getElementById('alerts-list');
        alertsList.innerHTML = '';
        
        data.slice(-10).reverse().forEach(alert => {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert-item ${alert.level} slide-in`;
            alertDiv.innerHTML = `
                <div class="alert-timestamp">${new Date(alert.timestamp).toLocaleString()}</div>
                <div class="alert-message">${alert.message}</div>
            `;
            alertsList.appendChild(alertDiv);
        });
        
        document.getElementById('alert-count').textContent = data.length;
    }
    
    updatePerformanceStats(data) {
        const avgStep = data && data.avg_step_time ? `${data.avg_step_time.toFixed(3)}s` : '0.000s';
        const avgEpoch = data && data.avg_epoch_time ? `${data.avg_epoch_time.toFixed(3)}s` : '0.000s';
        const avgThroughput = data && data.avg_throughput ? data.avg_throughput.toFixed(1) : '0.0';
        const minThroughput = data && data.min_throughput ? data.min_throughput.toFixed(1) : '0.0';

        document.getElementById('avg-step-time').textContent = avgStep;
        document.getElementById('avg-epoch-time').textContent = avgEpoch;
        document.getElementById('avg-throughput').textContent = avgThroughput;
        document.getElementById('min-throughput').textContent = minThroughput;
    }
    
    updateTrainingData(data) {
        if (!this.liveMode) {
            return;
        }
        // Add new data point to training chart
        const timeLabel = new Date(data.timestamp).toLocaleTimeString();
        
        this.trainingChart.data.labels.push(timeLabel);
        this.trainingChart.data.datasets[0].data.push(data.loss);
        this.trainingChart.data.datasets[1].data.push(data.learning_rate);
        this.trainingChart.data.datasets[2].data.push(data.throughput);
        
        // Keep only last 50 points
        if (this.trainingChart.data.labels.length > 50) {
            this.trainingChart.data.labels.shift();
            this.trainingChart.data.datasets.forEach(dataset => dataset.data.shift());
        }
        
        this.trainingChart.update('none');
    }
    
    updateSystemData(data) {
        if (!this.liveMode) {
            return;
        }
        // Add new data point to system chart
        const timeLabel = new Date(data.timestamp).toLocaleTimeString();
        
        this.systemChart.data.labels.push(timeLabel);
        this.systemChart.data.datasets[0].data.push(data.cpu_percent);
        this.systemChart.data.datasets[1].data.push(data.memory_percent);
        
        // Keep only last 50 points
        if (this.systemChart.data.labels.length > 50) {
            this.systemChart.data.labels.shift();
            this.systemChart.data.datasets.forEach(dataset => dataset.data.shift());
        }
        
        this.systemChart.update('none');
    }
    
    updateProgressData(data) {
        if (!this.liveMode) {
            return;
        }
        this.updateProgress(data);
    }
    
    addAlert(alert) {
        if (!this.liveMode) {
            return;
        }
        const alertsList = document.getElementById('alerts-list');
        
        // Remove "No alerts" message if present
        const noAlertsMsg = alertsList.querySelector('.text-muted');
        if (noAlertsMsg) {
            noAlertsMsg.remove();
        }
        
        // Create new alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert-item ${alert.level} slide-in`;
        alertDiv.innerHTML = `
            <div class="alert-timestamp">${new Date(alert.timestamp).toLocaleString()}</div>
            <div class="alert-message">${alert.message}</div>
        `;
        
        // Insert at the top
        alertsList.insertBefore(alertDiv, alertsList.firstChild);
        
        // Keep only last 10 alerts
        while (alertsList.children.length > 10) {
            alertsList.removeChild(alertsList.lastChild);
        }
        
        // Update alert count
        const currentCount = parseInt(document.getElementById('alert-count').textContent);
        document.getElementById('alert-count').textContent = currentCount + 1;
    }
    
    updateStatusIndicator(text, type) {
        const indicator = document.getElementById('status-indicator');
        indicator.textContent = text;
        indicator.className = `badge bg-${type} me-2`;
    }
    
    showLoading(show) {
        if (!this.loadingModal) {
            const modalEl = document.getElementById('loadingModal');
            if (!modalEl) {
                return;
            }
            this.loadingModal = new bootstrap.Modal(modalEl);
        }
        const modal = this.loadingModal;
        if (show) {
            modal.show();
        } else {
            modal.hide();
        }
    }
    
    showAlert(message, type) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at the top of the container
        const container = document.querySelector('.container-fluid');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
    
    async startMonitoring() {
        this.switchToLive(false);
        try {
            await fetch('/api/control/start', { method: 'POST' });
            this.showAlert('Monitoring started', 'success');
            this.refreshData();
        } catch (error) {
            this.showAlert('Error starting monitoring: ' + error.message, 'danger');
        }
    }
    
    async stopMonitoring() {
        this.switchToLive(false);
        try {
            await fetch('/api/control/stop', { method: 'POST' });
            this.showAlert('Monitoring stopped', 'info');
            this.refreshData();
        } catch (error) {
            this.showAlert('Error stopping monitoring: ' + error.message, 'danger');
        }
    }
    
    async resetMonitoring() {
        this.switchToLive(false);
        if (confirm('Are you sure you want to reset all monitoring data?')) {
            try {
                await fetch('/api/control/reset', { method: 'POST' });
                this.showAlert('Monitoring data reset', 'warning');
                this.refreshData();
            } catch (error) {
                this.showAlert('Error resetting monitoring: ' + error.message, 'danger');
            }
        }
    }
    
    async exportData() {
        this.switchToLive(false);
        try {
            const response = await fetch('/api/export/json');
            const data = await response.json();
            
            if (data.success) {
                this.showAlert('Data exported successfully', 'success');
                // You could add download functionality here
            } else {
                this.showAlert('Export failed: ' + data.error, 'danger');
            }
        } catch (error) {
            this.showAlert('Error exporting data: ' + error.message, 'danger');
        }
    }

    async loadExperiments() {
        try {
            const data = await this.fetchData('/api/experiments');
            this.experiments = data.experiments || [];
            this.populateExperimentSelector();
        } catch (error) {
            console.error('Failed to load experiments:', error);
        }
    }

    populateExperimentSelector() {
        const select = document.getElementById('experiment-select');
        if (!select) {
            return;
        }

        const previous = select.value;
        select.innerHTML = '';

        const liveOption = document.createElement('option');
        liveOption.value = '__live__';
        liveOption.textContent = 'Live Monitor';
        select.appendChild(liveOption);

        this.experiments.forEach((exp) => {
            const option = document.createElement('option');
            option.value = exp.id;
            const relativeDir = exp.relative_dir && exp.relative_dir !== '.' ? ` / ${exp.relative_dir}` : '';
            option.textContent = `${exp.project} / ${exp.run_id}${relativeDir}`;
            if (exp.updated_at) {
                option.dataset.updatedAt = exp.updated_at;
            }
            select.appendChild(option);
        });

        if (previous && Array.from(select.options).some(opt => opt.value === previous)) {
            select.value = previous;
        } else {
            select.value = '__live__';
            this.switchToLive(false);
        }
    }

    handleExperimentChange(value) {
        if (!value || value === '__live__') {
            this.switchToLive();
            return;
        }
        this.switchToExperiment(value);
        this.refreshData();
    }

    switchToLive(refresh = true) {
        this.liveMode = true;
        this.currentExperimentId = null;

        const select = document.getElementById('experiment-select');
        if (select && select.value !== '__live__') {
            select.value = '__live__';
        }

        if (this.isConnected) {
            this.updateStatusIndicator('Connected', 'success');
        } else {
            this.updateStatusIndicator('Disconnected', 'danger');
        }

        if (refresh) {
            this.refreshData();
        }
    }

    switchToExperiment(experimentId) {
        this.liveMode = false;
        this.currentExperimentId = experimentId;

        const select = document.getElementById('experiment-select');
        if (select && select.value !== experimentId) {
            select.value = experimentId;
        }

        const selected = this.experiments.find(exp => exp.id === experimentId);
        if (selected && selected.updated_at) {
            this.updateStatusIndicator(`Snapshot (${selected.updated_at})`, 'info');
        } else {
            this.updateStatusIndicator('Snapshot', 'info');
        }
    }

    async fetchExperimentSnapshot(experimentId) {
        try {
            const url = `/api/experiments/${this.encodeExperimentId(experimentId)}`;
            return await this.fetchData(url);
        } catch (error) {
            this.showAlert('Failed to load experiment: ' + error.message, 'danger');
            return null;
        }
    }

    encodeExperimentId(experimentId) {
        return experimentId
            .split('/')
            .map(encodeURIComponent)
            .join('/');
    }

    applyExperimentSnapshot(snapshot) {
        if (!snapshot) {
            return;
        }

        const { status, training_metrics, system_metrics, progress, alerts, performance } = snapshot;

        if (status) {
            const statusText = status.project
                ? `${status.project} / ${status.run_id}`
                : 'Snapshot';
            const monitorStatusEl = document.getElementById('monitor-status');
            if (monitorStatusEl) {
                monitorStatusEl.textContent = statusText;
            }

            const indicatorLabel = status.updated_at ? `Snapshot (${status.updated_at})` : 'Snapshot';
            this.updateStatusIndicator(indicatorLabel, 'info');

            const alertCountEl = document.getElementById('alert-count');
            if (alertCountEl) {
                alertCountEl.textContent = alerts ? alerts.length : 0;
            }
        }

        this.updateTrainingChart(training_metrics || []);
        this.updateSystemChart(system_metrics || []);
        this.updateProgress(progress || {});
        this.updateAlerts(alerts || []);
        this.updatePerformanceStats(performance || {});
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MonitorDashboard();
});


