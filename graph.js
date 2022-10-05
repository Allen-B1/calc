class Graph {
    /** @type {HTMLCanvasElement} */
    elem

    /** @type {CanvasRenderingContext2D} */
    ctx

    xmin
    xmax
    ymin
    ymax

    /** @type {{[id: string]: {func: (x: number[]) => Array<number | void>, color: string}}} */
    lines

    /** @type {{[id: string]: {points: [number, number][], color: string}}} */
    curves

    /** @param {HTMLElement} container */
    constructor(container) {
        this.elem = document.createElement("canvas");
        this.ctx = this.elem.getContext("2d");
        container.appendChild(this.elem);
        window.addEventListener("resize", () => {
            this.elem.width = container.clientWidth;
            this.elem.height = container.clientHeight;
            console.log("resize");
            console.log(container.clientWidth);
            this.draw();
        });
        this.elem.width = container.clientWidth;
        this.elem.height = container.clientHeight;

        this.lines = {};
        this.curves = {};

        this.xmin = -10;
        this.xmax = 10;
        this.ymin =  -10;
        this.ymax = 10;

        this.draw();
    }

    /** Coordinates -> pixels */
    transform(x, y) {
        let px = (x - this.xmin) * this.elem.width / (this.xmax - this.xmin);
        let py = (this.ymax - y) * this.elem.height / (this.ymax - this.ymin);
        return [px, py];
    }


    addLine(f, color) {
        let id = Math.random().toString(36).slice(2);
        this.lines[id] = {
            func: f,
            color: color
        };
        return id;
    }

    removeLine(id) {
        delete this.lines[id];
    }

    removeAll() {
        this.lines = {};
        this.curves = {};
    }

    addCurve(points, color) {
        let id = Math.random().toString(36).slice(2);
        this.curves[id] = {
            points: points,
            color: color
        };
        return id;
    }

    draw() {
        this.ctx.fillStyle = "#fff";
        this.ctx.fillRect(0, 0, this.elem.width, this.elem.height);

        // draw axes
        this.ctx.strokeStyle = "#000";
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        for (let x = Math.ceil(this.xmin); x < Math.floor(this.xmax); x++) {
            let [px, _] = this.transform(x, 0);
            this.ctx.moveTo(px, 0);
            this.ctx.lineTo(px, this.elem.height);
        }
        for (let y = Math.ceil(this.ymin); y < Math.floor(this.ymax); y++) {
            let [_, py] = this.transform(0, y);
            this.ctx.moveTo(0, py);
            this.ctx.lineTo(this.elem.width, py);
        }
        this.ctx.stroke();

        this.ctx.strokeStyle = "#000";
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        let [ox, oy] = this.transform(0, 0);
        this.ctx.moveTo(ox, 0);
        this.ctx.lineTo(ox, this.elem.height);
        this.ctx.moveTo(0, oy);
        this.ctx.lineTo(this.elem.width, oy);
        this.ctx.stroke();

        // draw functions
        for (let id in this.lines) {
            let line = this.lines[id];
            let x = [];
            for (let s = this.xmin; s < this.xmax; s += (this.xmax - this.xmin)/1024) {
                x.push(s);
            }
            let y = line.func(x);
            
            this.ctx.strokeStyle = line.color;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            let first = true;
            for (let i = 0; i < x.length; i++) {
                if (y[i] == null) {
                    first = true;
                    continue;
                }
                let coords = this.transform(x[i], y[i]);
                if (first) {
                    this.ctx.moveTo(...coords);
                } else {
                    this.ctx.lineTo(...coords);
                }
                first = false;
            }
            this.ctx.stroke();
        }

        for (let id in this.curves) {
            let curve = this.curves[id];
            this.ctx.strokeStyle = curve.color;
            this.ctx.lineWidth = 2;

            this.ctx.beginPath();
            let point = curve.points[0];
            this.ctx.moveTo(...this.transform(point[0], point[1]));
            for (let i = 1; i < curve.points.length; i++) {
                let point = curve.points[i];
                this.ctx.lineTo(...this.transform(point[0], point[1]));
            }
            this.ctx.stroke();
        }
    }
}
